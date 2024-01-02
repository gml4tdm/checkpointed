from __future__ import annotations

from .handle import PipelineStepHandle
from .step import PipelineStep


class Instruction:
    pass


class Start(Instruction):

    def __init__(self,
                 step: PipelineStepHandle,
                 factory: type[PipelineStep],
                 inputs: list[tuple[PipelineStepHandle, type[PipelineStep], str]]):
        self.step = step
        self.factory = factory
        self.inputs = inputs


class Sync(Instruction):

    def __init__(self, steps: list[PipelineStepHandle], then: list[Start]):
        self.steps = frozenset(steps)
        self.then = then
