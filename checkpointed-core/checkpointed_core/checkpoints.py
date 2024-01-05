from __future__ import annotations

import collections
import itertools
import logging
import typing

from .step import PipelineStep
from .handle import PipelineStepHandle


class CheckpointGraph:

    def __init__(self,
                 inputs: set[PipelineStepHandle],
                 vertices: set[PipelineStepHandle],
                 factories: dict[PipelineStepHandle, type[PipelineStep]],
                 connections: dict[PipelineStepHandle, list[PipelineStepHandle]],
                 connection_types: dict[tuple[PipelineStepHandle, PipelineStepHandle], str],
                 config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                 *,
                 logger: logging.Logger | None = None):
        if logger is None:
            self._logger = logging.getLogger(__name__)
        else:
            self._logger = logger
        self.inputs = inputs
        self.vertices = vertices
        self.factories = {
            handle: factory.__name__ for handle, factory in factories.items()
        }
        self.connections = connections
        self.connection_types = connection_types
        self.config_by_step = config_by_step
        # Pre-computed auxiliary attributes
        self.handles_by_factory = collections.defaultdict(set)
        for handle, factory in self.factories.items():
            self.handles_by_factory[factory].add(handle)
        self.incoming_per_handle = collections.defaultdict(dict)
        for source, targets in self.connections.items():
            for target in targets:
                self.incoming_per_handle[target][self.connection_types[(source, target)]] = source

    def _log_graphs(self, other: CheckpointGraph):
        self._logger.debug('Computing cacheable nodes...')
        self._logger.debug('Old graph description:')
        self._log_graph_properties(self, self._logger)
        self._logger.debug('New graph description:')
        self._log_graph_properties(other, self._logger)

    @staticmethod
    def _log_graph_properties(graph: CheckpointGraph, logger: logging.Logger):
        logger.debug(' * Inputs: %s', graph.inputs)
        logger.debug(' * Vertices: %s', graph.vertices)
        logger.debug(' * Factories: %s', graph.factories)
        logger.debug(' * Connections: %s', graph.connections)
        logger.debug(' * Configs: %s', graph.config_by_step)

    def update_caching_candidates(self,
                                  candidates: dict[PipelineStepHandle, PipelineStepHandle],
                                  valid_checkpoints: set[PipelineStepHandle]) -> dict[PipelineStepHandle, PipelineStepHandle]:
        self._logger.info('Valid checkpoints: %s', ', '.join(map(str, sorted(valid_checkpoints))))
        cacheable = {
            (x, y)
            for x, y in candidates.items()
            if x in self.inputs and y in valid_checkpoints
        }
        while True:
            additions = {
                (x, y)
                for x, y in candidates.items()
                if (
                        (x, y) not in cacheable and
                        y in valid_checkpoints and
                        self._inputs_are_valid(x, y, candidates, cacheable)
                )
            }
            if not additions:
                break
            cacheable |= additions
        self._logger.info(f'Found {len(cacheable)} cacheable steps...')
        self._logger.info(f'Caching mapping:')
        cacheable = dict(cacheable)
        for new in sorted(cacheable):
            self._logger.info(f' * {cacheable[new]} -> {new}')
        return cacheable

    def _inputs_are_valid(self,
                          x, _y,
                          mapping: dict[PipelineStepHandle, PipelineStepHandle],
                          cacheable: dict[PipelineStepHandle, PipelineStepHandle]) -> bool:
        for z in self.incoming_per_handle[x].values():
            if (z, mapping[z]) not in cacheable:
                return False
        return True

    def get_caching_candidates(self,
                               other: CheckpointGraph) -> dict[PipelineStepHandle, PipelineStepHandle]:
        """Starting from all input nodes, determine all nodes that
        have equivalent nodes in the other graph.
        Here, equivalent means that:

            1) They have the same factory
            2) They have the same config
            3) They have the same _incoming_ connections

        Note that we are not interested about outgoing connections.
        This is because this method is only meant to be used
        to determine what steps can be safely loaded from checkpoints.
        For this, for any given state, we only need to check
        preceding states.

        There are a number of challenges with a general implementation
        of this caching algorithm. In particular, if there are two
        identical input steps in a given graph, we have to try
        all possible isomorphisms.
        """
        self._log_graphs(other)
        best = None
        for lineup in self._generate_equivalent_vertices(other):
            # lineup is a list of (x, y) pairs, where x and y
            # are handles in the new and old graph, respectively.
            # For every pair (x, y), the corresponding factories
            # are the same, and they have the same config.
            #
            # What we need to check is to what extent this mapping
            # matches the graph structure of the old graph.
            #
            # Every lineup is scored according to the number of
            # steps that can be cached.
            #
            # In the end, we will return the lineup which results
            # in the largest amount of cached steps.
            mapping = self._compute_cacheable_steps(lineup, other)
            if best is None:
                best = mapping
            else:
                best = max(best, mapping, key=len)
        # Return a dictionary containing all steps which can be
        # cached, and mapping them to their handles in the
        # old graph.
        if best is None:
            best = {}
        self._logger.info(f'Caching up to {len(best)} steps...')
        self._logger.info(f'Proposed caching mapping:')
        cacheable = dict(best)
        for new in sorted(cacheable):
            self._logger.info(f' * {cacheable[new]} -> {new}')
        return cacheable

    def _generate_equivalent_vertices(self, other: CheckpointGraph):
        # First, for every node, collect a set of all possible
        # matching nodes in the other graph.
        pairings_per_node = collections.defaultdict(list)
        for factory, handles in self.handles_by_factory.items():
            for x in handles:
                for y in other.handles_by_factory[factory]:
                    if self.config_by_step[x] == other.config_by_step[y]:
                        self._logger.info(f'Found possible isomorphic nodes: {x} and {y}')
                        pairings_per_node[x].append((x, y))
        # Now, return every possible combination of pairings.
        number_of_lineups = 1
        for pair in pairings_per_node.values():
            number_of_lineups *= len(pair)
        self._logger.info(f'Checking {number_of_lineups} possible lineups...')
        yield from itertools.product(*pairings_per_node.values())

    def _compute_cacheable_steps(self,
                                 lineup: list[tuple[PipelineStepHandle, PipelineStepHandle]],
                                 other: CheckpointGraph) -> set[tuple[PipelineStepHandle, PipelineStepHandle]]:
        cacheable = {
            (x, y)
            for x, y in lineup
            if (
                    x in self.inputs and
                    y in other.inputs
            )
        }
        while True:
            additions = {
                (x, y)
                for x, y in lineup
                if (
                        (x, y) not in cacheable and
                        self._align_incoming(x, y, cacheable, other)
                )
            }
            if not additions:
                break
            cacheable |= additions
        return cacheable

    def _align_incoming(self,
                        x: PipelineStepHandle,
                        y: PipelineStepHandle,
                        cacheable: set[tuple[PipelineStepHandle, PipelineStepHandle]],
                        other: CheckpointGraph) -> bool:
        if len(self.incoming_per_handle[x]) != len(other.incoming_per_handle[y]):
            return False
        if set(self.incoming_per_handle[x]) != set(other.incoming_per_handle[y]):
            return False
        for key in self.incoming_per_handle[x]:
            if self.incoming_per_handle[x][key] != other.incoming_per_handle[y][key]:
                return False
            if (self.incoming_per_handle[x][key], self.incoming_per_handle[x][key]) not in cacheable:
                return False
        return True
