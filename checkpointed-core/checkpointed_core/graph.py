import dataclasses

from .handle import PipelineStepHandle
from .step import PipelineStep

__all__ = [
    'PipelineNode',
    'PipelineConnection',
    'PipelineGraph',
]


@dataclasses.dataclass(frozen=True)
class PipelineNode:
    name: str | None
    handle: PipelineStepHandle
    factory: type[PipelineStep]
    is_input: bool
    is_output: bool
    output_filename: str | None = None


@dataclasses.dataclass(frozen=True)
class PipelineConnection:
    source: PipelineStepHandle
    target: PipelineStepHandle
    label: str


@dataclasses.dataclass(frozen=True)
class PipelineGraph:
    vertices: list[PipelineNode]
    edges: list[PipelineConnection]
