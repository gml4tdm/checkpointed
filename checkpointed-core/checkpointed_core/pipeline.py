from __future__ import annotations

import collections
import graphlib
import typing

from .graph import *
from .handle import PipelineStepHandle
from .step import PipelineStep
from .instructions import Instruction, Start, Sync
from .plan import ExecutionPlan

__all__ = ['Pipeline']


class Pipeline:

    def __init__(self, name: str):
        self.name = name
        self._nodes: dict[PipelineStepHandle, PipelineNode] = {}
        self._edges: dict[PipelineStepHandle, dict[str, PipelineStepHandle]] = {}

    def as_graph(self) -> PipelineGraph:
        return PipelineGraph(
            vertices=list(self._nodes.values()),
            edges=[
                PipelineConnection(source, target, label)
                for source, connections in self._edges.items()
                for label, target in connections.items()
            ]
        )

    def add_step(self,
                 factory: type[PipelineStep],
                 name: None | str = None) -> PipelineStepHandle:
        return self.add_node(
            factory=factory,
            name=name,
            is_source=False,
            is_sink=False
        )

    def add_source(self,
                   factory: type[PipelineStep],
                   name: None | str = None) -> PipelineStepHandle:
        return self.add_node(
            factory=factory,
            name=name,
            is_source=True,
            is_sink=False
        )

    def add_sink(self,
                 factory: type[PipelineStep],
                 filename: str,
                 name: None | str = None) -> PipelineStepHandle:
        return self.add_node(
            factory=factory,
            name=name,
            is_source=False,
            is_sink=True,
            filename=filename
        )

    def add_source_sink(self,
                        factory: type[PipelineStep],
                        filename: str,
                        name: None | str = None) -> PipelineStepHandle:
        return self.add_node(
            factory=factory,
            name=name,
            is_source=True,
            is_sink=True,
            filename=filename
        )

    def add_node(self, *,
                 factory: type[PipelineStep],
                 is_source: bool,
                 is_sink: bool,
                 filename: str | None = None,
                 name: str | None = None) -> PipelineStepHandle:
        if is_sink and filename is None:
            raise ValueError("filename must be specified for sink nodes")
        handle = PipelineStepHandle(len(self._nodes), name)
        node = PipelineNode(
            name=name,
            factory=factory,
            handle=handle,
            is_input=is_source,
            is_output=is_sink,
            output_filename=filename
        )
        self._nodes[handle] = node
        return handle

    def connect(self,
                source: PipelineStepHandle,
                sink: PipelineStepHandle,
                label: str, *,
                streaming=False):
        if source not in self._nodes:
            raise ValueError(f"source node {source} not found")
        if sink not in self._nodes:
            raise ValueError(f"sink node {sink} not found")
        if self._nodes[sink].is_input:
            raise ValueError(f"sink node {sink} is an input node")
        if source == sink:
            raise ValueError(f"source and sink nodes cannot be the same")
        if streaming:
            supported = self._nodes[sink].factory.supported_streamed_inputs()
        else:
            supported = self._nodes[sink].factory.supported_inputs()
        if label not in supported and ... not in supported:
            raise ValueError(f"sink node {sink} does not support input type {label}")
        if not issubclass(self._nodes[source].factory, supported[label]):
            if label is ...:
                raise TypeError(
                    f'Sink node of type {self._nodes[sink].factory.__name__}) wildcard'
                    f'connection (...) does not support input of type '
                    f'{self._nodes[source].factory.__name__}'
                )
            raise TypeError(
                f'Sink node of type {self._nodes[sink].factory.__name__}) connection '
                f'{label!r} does not support input of type '
                f'{self._nodes[source].factory.__name__}'
            )
        if source in self._edges:
            if label in self._edges[source]:
                raise ValueError(f"connection {source} -> {sink} of type {label} already exists")
        else:
            self._edges[source] = {}
        self._edges[source][label] = sink

    def build(self,
              config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]]) -> ExecutionPlan:
        self._check_cycles()
        self._check_reachability()
        self._check_incoming_connections()
        self._check_source_sink_constraints()
        instructions = self._build_instruction_list()
        return ExecutionPlan(
            name=self.name,
            instructions=instructions,
            config_by_step=config_by_step,
            graph=self.as_graph()
        )

    def _build_instruction_list(self) -> list[Instruction]:
        incoming_per_node = self._get_incoming_by_node()
        steps_per_group = collections.defaultdict(set)
        for handle, incoming in incoming_per_node.items():
            steps_per_group[tuple(sorted(incoming))].add(handle)
        label_by_connection = {}
        for source, connections in self._edges.items():
            for label, target in connections.items():
                label_by_connection[(source, target)] = label
        instructions = []
        for dependencies, handles in steps_per_group:
            instructions.append(
                Sync(
                    list(dependencies),
                    [
                        Start(
                            node.handle,
                            node.factory,
                            [
                                (
                                    handle,
                                    self._nodes[handle].factory,
                                    label_by_connection[(handle, node.handle)]
                                )
                                for handle in incoming_per_node[node.handle]
                            ]
                        )
                        for node in map(lambda h: self._nodes[h], handles)
                    ]
                )
            )
        return instructions

    def _get_incoming_by_node(self) -> dict[PipelineStepHandle, set[PipelineStepHandle]]:
        incoming_by_node = collections.defaultdict(set)
        for source, connections in self._edges.items():
            for label, target in connections.items():
                incoming_by_node[target].add(source)
        return dict(incoming_by_node)

    def _check_incoming_connections(self):
        incoming_by_node = self._get_incoming_by_node()
        for handle, incoming in incoming_by_node.items():
            expected = set(self._nodes[handle].factory.supported_inputs())
            missing = expected - incoming
            missing.discard(...)
            if missing:
                raise ValueError(
                    f"missing incoming connections for step {handle}: {sorted(missing)}"
                )

    def _check_source_sink_constraints(self):
        incoming_by_node = self._get_incoming_by_node()
        for handle, node in self._nodes.items():
            if node.is_input and handle not in self._edges:
                if not node.is_output:
                    raise ValueError(
                        f"Input step {handle} has no outgoing connections."
                    )
            elif node.is_output and node not in incoming_by_node:
                raise ValueError(
                    f"Output step {handle} has no incoming connections."
                )
            elif handle not in self._edges:
                raise ValueError(
                    f"Regular step {handle} has no outgoing connections."
                )
            elif handle not in incoming_by_node:
                raise ValueError(
                    f"Regular step {handle} has no incoming connections."
                )

    def _check_reachability(self):
        reachable = set()
        stack = [
            node.handle for node in self._nodes.values() if node.is_input
        ]
        while stack:
            handle = stack.pop()
            if handle not in reachable:
                reachable.add(handle)
                stack.extend(self._edges[handle].values())
        missing = set(self._nodes.keys()) - reachable
        if missing:
            raise ValueError(f"Unreachable steps in pipeline: {sorted(missing)}")

    def _check_cycles(self):
        graph = {
            source: set(connections.values())
            for source, connections in self._edges.items()
        }
        try:
            graphlib.TopologicalSorter(graph).static_order()
        except graphlib.CycleError:
            raise ValueError("pipeline contains a cycle")
