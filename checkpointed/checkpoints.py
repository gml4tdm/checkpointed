from __future__ import annotations

import collections
import itertools
import logging
import typing

from checkpointed import PipelineStep
from checkpointed.handle import PipelineStepHandle


class CheckpointManager:
    pass


class CheckpointGraph:

    def __init__(self,
                 inputs: set[PipelineStepHandle],
                 vertices: set[PipelineStepHandle],
                 factories: dict[PipelineStepHandle, type[PipelineStep]],
                 connections: dict[PipelineStepHandle, list[PipelineStepHandle]],
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
        self.config_by_step = config_by_step
        # Pre-computed auxiliary attributes
        self.handles_by_factory = collections.defaultdict(set)
        for handle, factory in self.factories.items():
            self.handles_by_factory[factory].add(handle)
        self.incoming_per_handle = collections.defaultdict(set)
        for source, targets in self.connections.items():
            for target in targets:
                self.incoming_per_handle[target].add(source)

    def _log_graphs(self, other: CheckpointGraph):
        self._logger.debug('Computing cacheable nodes...')
        self._logger.debug('Old graph description:')
        self._logger.debug(' * Inputs: {}', other.inputs)
        self._logger.debug(' * Vertices: {}', other.vertices)
        self._logger.debug(' * Factories: {}', other.factories)
        self._logger.debug(' * Connections: {}', other.connections)
        self._logger.debug(' * Configs: {}', other.config_by_step)
        self._logger.debug('New graph description:')
        self._logger.debug(' * Inputs: {}', self.inputs)
        self._logger.debug(' * Vertices: {}', self.vertices)
        self._logger.debug(' * Factories: {}', self.factories)
        self._logger.debug(' * Connections: {}', self.connections)
        self._logger.debug(' * Configs: {}', self.config_by_step)

    def get_largest_isomorphic_prefix(self, other: CheckpointGraph):
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
            best = max(best, mapping, key=len)
        return best

    def _generate_equivalent_vertices(self, other: CheckpointGraph):
        # First, for every node, collect a set of all possible
        # matching nodes in the other graph.
        factories = {self.factories[h] for h in self.inputs}
        pairings_per_node = collections.defaultdict(list)
        for factory in factories:
            for x in self.handles_by_factory[factory]:
                for y in other.handles_by_factory[factory]:
                    if self.config_by_step[x] == other.config_by_step[y]:
                        self._logger.debug('Found possible isomorphic nodes: {} and {}', x, y)
                        pairings_per_node[x].append((x, y))
        # Now, return every possible combination of pairings.
        number_of_lineups = 1
        for pair in pairings_per_node.values():
            number_of_lineups *= len(pair)
        self._logger.debug(f'Checking {number_of_lineups} possible lineups...')
        yield from itertools.product(*pairings_per_node.values())

    def _compute_cacheable_steps(self,
                                 lineup: list[tuple[PipelineStepHandle, PipelineStepHandle]],
                                 other: CheckpointGraph) -> int:
        cacheable = {
            (x, y)
            for x, y in lineup
            if x in self.inputs and y in other.inputs
        }
        while True:
            additions = {
                (x, y)
                for x, y in lineup
                if (x, y) not in cacheable and self._align_incoming(x, y, cacheable, other)
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
        for p, q in zip(self.incoming_per_handle[x], other.incoming_per_handle[y]):
            if (p, q) not in cacheable:
                return False
        return True
