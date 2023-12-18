from __future__ import annotations

from .handle import PipelineStepHandle
from .step import PipelineStep


class Instruction:
    pass


class Start(Instruction):

    def __init__(self, step: PipelineStepHandle, factory: type[PipelineStep]):
        self.step = step
        self.factory = factory


class Sync(Instruction):

    def __init__(self, steps: list[PipelineStepHandle], then: list[Start]):
        self.steps = frozenset(steps)
        self.then = then
