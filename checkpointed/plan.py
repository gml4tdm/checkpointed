from __future__ import annotations

import logging
import os

from .instructions import Instruction
from .handle import PipelineStepHandle
from . import executor
from . import store


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

    def execute(self,
                config_by_step: dict[PipelineStepHandle, dict], *,
                output_directory='',
                checkpoint_directory='',
                logger: logging.Logger | None = None):
        if logger is None:
            logger = logging.getLogger(self._name)
            logger.addHandler(logging.NullHandler())
        result_store = store.ResultStore(
            output_directory=os.path.join(output_directory, self._name),
            checkpoint_directory=os.path.join(checkpoint_directory, self._name),
            file_by_step=self._output_files,
            output_steps=self._outputs,
            max_size=0
        )
        task_executor = executor.TaskExecutor(self._instructions)
        task_executor.run(result_store, config_by_step, logger)
