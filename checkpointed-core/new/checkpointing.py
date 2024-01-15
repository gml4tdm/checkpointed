from __future__ import annotations

import collections
import itertools
import logging
import math
import typing

from .graph import PipelineGraph
from .handle import PipelineStepHandle


class CheckpointGraph:

    def __init__(self,
                 graph: PipelineGraph,
                 config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]]):
        self._input_labels_per_node = {
            node.handle: set(node.factory.supported_inputs().keys())
            for node in graph.vertices
        }
        self._factories = {
            node.handle: node.factory.__name__
            for node in graph.vertices
        }
        self._outputs_per_node = collections.defaultdict(list)
        for connection in graph.edges:
            self._outputs_per_node[connection.source].append(
                (connection.target, connection.label)
            )
        self._inputs_per_node = {}
        for connection in graph.edges:
            self._inputs_per_node[(connection.target, connection.label)] = connection.source
        self._input_nodes = {
            node.handle for node in graph.vertices if node.is_input
        }
        self._config_by_step = config_by_step

    @property
    def _handles_by_factory(self) -> dict[str, list[PipelineStepHandle]]:
        handles_by_factory = collections.defaultdict(list)
        for handle, factory in self._factories.items():
            handles_by_factory[factory].append(handle)
        return handles_by_factory

    def compute_checkpoint_mapping(
            self,
            old: CheckpointGraph,
            logger: logging.Logger) -> dict[PipelineStepHandle, PipelineStepHandle]:
        return self._compute_best_matchup(old, logger)

    def _compute_best_matchup(self,
                              old: CheckpointGraph,
                              logger: logging.Logger) -> dict[PipelineStepHandle, PipelineStepHandle]:
        best = {}
        for matchup in self._compute_equivalent_nodes(old, logger):
            mapping = self._compute_caching_mapping_from_matchup(matchup,
                                                                 old,
                                                                 logger)
            best = max(best, mapping, key=len)
        return best

    def _compute_equivalent_nodes(self,
                                  old: CheckpointGraph,
                                  logger: logging.Logger):
        matchups_per_node = collections.defaultdict(list)
        for factory, handles in self._handles_by_factory.items():
            for new_handle in handles:
                for old_handle in old._handles_by_factory[factory]:
                    if self._config_by_step[new_handle] != old._config_by_step[old_handle]:
                        continue
                    logger.info(f'Found possibly matching nodes: {old_handle} -> {new_handle}')
                    matchups_per_node[new_handle].append((new_handle, old_handle))
        number_of_matchups = math.prod(len(x) for x in matchups_per_node.values())
        logger.info(f'Possible number of node matchups: {number_of_matchups}')
        yield from itertools.product(*matchups_per_node.values())

    def _compute_caching_mapping_from_matchup(
            self,
            matchup: list[tuple[PipelineStepHandle, PipelineStepHandle]],
            old: CheckpointGraph,
            #valid_checkpoints: set[PipelineStepHandle],
            _logger: logging.Logger):
        cacheable = {
            (x, y)
            for x, y in matchup
            if (
                x in self._input_nodes and
                y in old._input_nodes
                #y in valid_checkpoints
            )
        }
        while True:
            additions = {
                (x, y)
                for x, y in matchup
                if (
                    (x, y) not in cacheable and
                    #y in valid_checkpoints and
                    self._check_input_compatibility(x, y, old, cacheable)
                )
            }
            if not additions:
                break
            cacheable |= additions
        return dict(cacheable)

    def _check_input_compatibility(self,
                                   x: PipelineStepHandle,
                                   y: PipelineStepHandle,
                                   old: CheckpointGraph,
                                   cacheable: set[tuple[PipelineStepHandle, PipelineStepHandle]]) -> bool:
        if self._input_labels_per_node[x] != self._input_labels_per_node[y]:
            return False
        for key in self._input_labels_per_node[x]:
            p = self._inputs_per_node[(x, key)]
            q = old._inputs_per_node[(y, key)]
            if (p, q) not in cacheable:
                return False
        return True


