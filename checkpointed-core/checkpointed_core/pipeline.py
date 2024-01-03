from __future__ import annotations
import collections
import graphlib
import itertools
import logging
import typing

from . import step as _step
from .checkpoints import CheckpointGraph
from .handle import PipelineStepHandle
from .instructions import Instruction, Start, Sync
from .plan import ExecutionPlan


def _powerset(x):
    return itertools.chain.from_iterable(
        itertools.combinations(x, r) for r in range(1, len(x)+1)
    )


class Pipeline:

    def __init__(self, name: str):
        self._name = name
        self._steps: dict[PipelineStepHandle, type[_step.PipelineStep]] = {}
        self._connections = collections.defaultdict(list)
        self._connections_types = {}
        self._inputs = set()
        self._outputs = set()
        self._output_files = {}

    def add_source(self,
                   factory: type[_step.PipelineStep], *,
                   is_sink=False,
                   filename: str = '',
                   name: None | str = None) -> PipelineStepHandle:
        handle = self.add_step(factory, name=name)
        self._inputs.add(handle)
        if is_sink:
            if not filename:
                raise ValueError(f"Filename must be provided for sink")
            self._outputs.add(handle)
            self._output_files[handle] = filename
        return handle

    def add_sink(self
                 , factory: type[_step.PipelineStep], *,
                 filename: str,
                 name: None | str = None) -> PipelineStepHandle:
        handle = self.add_step(factory, name=name)
        self._outputs.add(handle)
        self._output_files[handle] = filename
        return handle

    def add_step(self,
                 factory: type[_step.PipelineStep], *,
                 name: None | str = None) -> PipelineStepHandle:
        handle = PipelineStepHandle(len(self._steps), name)
        self._steps[handle] = factory
        return handle

    def connect(self, source: PipelineStepHandle, sink: PipelineStepHandle, label: str):
        if source not in self._steps:
            raise ValueError(f"Source step {source} not found in pipeline {self._name}")
        if sink not in self._steps:
            raise ValueError(f"Sink step {sink} not found in pipeline {self._name}")
        if sink in self._connections[source]:
            raise ValueError(f"Connection already exists between {source} and {sink}")
        if sink in self._inputs:
            raise ValueError(f"Cannot use an input step ({sink}) as sink")
        if sink in self._connections[source]:
            raise ValueError(f"Cannot have multiple connections between source {source} and sink {sink}")
        if source == sink:
            raise ValueError(f"Cannot connect step {source} to itself")
        if not self._steps[sink].supports_step_as_input(self._steps[source], label):
            raise ValueError(f"Cannot connect {source} to {sink} (unsupported input for label {label})")
        assert (
                label in self._steps[sink].get_input_labels() or
                ... in self._steps[sink].get_input_labels()
        )
        self._connections[source].append(sink)
        self._connections_types[(source, sink)] = label

    def build(self,
              config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]], *,
              logger: logging.Logger | None = None):
        self._check_connection_constraints()
        self._check_source_sink_constraints()
        self._check_reachability_constraints()
        self._check_cycle_constraints()
        instructions = self._build_execution_plan()
        return ExecutionPlan(
            name=self._name,
            directory=self._name,
            instructions=instructions,
            output_steps=frozenset(self._outputs),
            output_files=self._output_files,
            config_by_step=config_by_step,
            logger=logger,
            graph=CheckpointGraph(
                inputs=self._inputs,
                vertices=set(self._steps),
                connections=self._connections,
                connection_types=self._connections_types,
                factories=self._steps,
                config_by_step=config_by_step,
                logger=logger
            )
        )

    def _build_execution_plan(self) -> list[Instruction]:
        dependencies_per_step = collections.defaultdict(set)
        for source, connections in self._connections.items():
            for sink in connections:
                dependencies_per_step[sink].add(source)
        plan_per_dependency_group = collections.defaultdict(set)
        instructions = []
        for handle in self._inputs:
            instructions.append(
                Start(
                    handle,
                    self._steps[handle],
                    [(h, self._steps[h], self._connections_types[(h, handle)])
                     for h in dependencies_per_step[handle]]
                )
            )
        for handle, dependencies in dependencies_per_step.items():
            if dependencies:
                key = tuple(sorted(dependencies, key=lambda h: h.get_raw_identifier()))
                plan_per_dependency_group[key].add(handle)
        for requirements, handles in plan_per_dependency_group.items():
            instructions.append(
                Sync(
                    list(requirements),
                    [
                        Start(
                            handle,
                            self._steps[handle],
                            [(h, self._steps[h], self._connections_types[(h, handle)])
                             for h in dependencies_per_step[handle]]
                        )
                        for handle in handles
                    ]
                )
            )
        return instructions

    def _check_connection_constraints(self):
        accepting_varargs = {
            handle
            for handle, factory in self._steps.items()
            if ... in factory.get_input_labels()
        }
        missing = {
            handle: set(factory.get_input_labels())
            for handle, factory in self._steps.items()
        }
        for source, sinks in self._connections.items():
            for sink in sinks:
                label = self._connections_types[(source, sink)]
                if label in missing[sink]:
                    missing[sink].remove(label)
                elif sink not in accepting_varargs:
                    raise ValueError(
                        f"Cannot connect {source} to {sink} (unsupported input for label {label})"
                    )
        missing_connections = ' | '.join(
            f"{handle} ({', '.join(missing[handle])})"
            for handle in missing
            if missing[handle]
        )
        if missing_connections:
            raise ValueError(f"Missing connections: {missing_connections}")

    def _check_source_sink_constraints(self):
        for handle in self._steps:
            cond = (
                    handle in self._inputs
                    or handle in self._outputs
                    or (self._is_source(handle) and self._is_sink(handle))
            )
            if not cond:
                raise ValueError('Found non-boundary step which has not both a source and a sink')
        for handle in self._inputs:
            if handle not in self._connections and handle not in self._outputs:
                raise ValueError('Found input step without any connections')
        for handle in self._outputs:
            if not self._is_sink(handle) and handle not in self._inputs:
                raise ValueError('Found output step without any connections')

    def _check_reachability_constraints(self):
        reachable = set()
        stack = list(self._inputs)
        while stack:
            handle = stack.pop()
            if handle not in reachable:
                reachable.add(handle)
                stack.extend(self._connections[handle])
        remainder = set(self._steps.keys()) - reachable
        if remainder:
            raise ValueError('Found unreachable steps')

    def _check_cycle_constraints(self):
        sorter = graphlib.TopologicalSorter(self._connections)
        try:
            sorter.static_order()
        except graphlib.CycleError:
            raise ValueError('Found cycle in pipeline')

    def _is_sink(self, handle: PipelineStepHandle):
        return any(handle in sinks for sinks in self._connections.values())

    def _is_source(self, handle: PipelineStepHandle):
        return handle in self._connections
