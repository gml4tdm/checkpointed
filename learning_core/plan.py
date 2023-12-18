from __future__ import annotations

import os
import typing

from learning_core.instructions import Instruction
from learning_core.handle import PipelineStepHandle


class ExecutionPlan:

    def __init__(self,
                 name: str,
                 directory: str,
                 instructions: list[Instruction],
                 output_steps: frozenset[PipelineStepHandle],
                 output_files: dict[PipelineStepHandle, str]):
        self._name = name
        self._instructions = instructions
        self._output_files = output_files
        self._directory = directory
        self._outputs = output_steps
        self._loaded_data = {}  # type: dict[PipelineStepHandle, typing.Any]

    def execute(self, configs: dict[PipelineStepHandle, dict]):
        os.makedirs(self._directory, exist_ok=True)
        for handle in self._order:
            if not self._is_checkpointed(handle):
                if handle in configs:
                    result = self._steps[handle].execute(config=configs[handle])
                else:
                    result = self._steps[handle].execute(config=None)
                if handle in self._outputs:
                    self._save_user_result(handle, result)
                self._checkpoint(handle, result)
        return

    def _save_user_result(self, handle: PipelineStepHandle, result):
        path = os.path.join(self._directory, 'output', self._output_files[handle])
        self._steps[handle].save_result(path)

    def _is_checkpointed(self, handle: PipelineStepHandle) -> bool:
        return os.path.exists(self._get_checkpoint_file(handle))

    def _checkpoint(self, handle: PipelineStepHandle, result):
        self._steps[handle].load_result(self._get_checkpoint_file(handle), result)

    def _load_checkpoint(self, handle: PipelineStepHandle) -> typing.Any:
        return self._steps[handle].load_result(self._get_checkpoint_file(handle), None)

    def _get_checkpoint_file(self, handle: PipelineStepHandle) -> str:
        path = os.path.join(self._directory, 'checkpoints', str(handle.get_raw_identifier()))
        return path


