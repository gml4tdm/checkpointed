from __future__ import annotations

import logging
import os
import typing

from .checkpoints import CheckpointGraph
from .instructions import Instruction
from .handle import PipelineStepHandle
from .step import PipelineStep
from . import executor
from . import store


class ExecutionPlan:

    def __init__(self,
                 name: str,
                 directory: str,
                 instructions: list[Instruction],
                 steps: dict[PipelineStepHandle, type[PipelineStep]],
                 output_steps: frozenset[PipelineStepHandle],
                 output_files: dict[PipelineStepHandle, str],
                 graph: CheckpointGraph,
                 config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                 logger: logging.Logger | None = None):
        self._name = name
        self._instructions = instructions
        self._output_files = output_files
        self._directory = directory
        self._outputs = output_steps
        self._graph = graph
        self._config_by_step = config_by_step
        self._steps = steps
        if logger is None:
            logger = logging.getLogger(self._name)
            logger.addHandler(logging.NullHandler())
        self._logger = logger

    def execute(self, *,
                output_directory='',
                checkpoint_directory='',
                __return_values: set[PipelineStepHandle] | None = None):
        result_store = store.ResultStore(
            output_directory=os.path.join(output_directory, self._name),
            checkpoint_directory=os.path.join(checkpoint_directory, self._name),
            file_by_step=self._output_files,
            factories_by_step=self._steps,
            output_steps=self._outputs,
            max_size=0,
            graph=self._graph,
            logger=self._logger
        )
        task_executor = executor.TaskExecutor(self._instructions)
        task_executor.run(result_store, self._config_by_step, self._logger)
        if __return_values is not None:
            return {step: result_store.retrieve(step, self._steps[step])
                    for step in __return_values}

    async def execute_async(self, *,
                            output_directory='',
                            checkpoint_directory='',
                            __return_values=None):
        result_store = store.ResultStore(
            output_directory=os.path.join(output_directory, self._name),
            checkpoint_directory=os.path.join(checkpoint_directory, self._name),
            file_by_step=self._output_files,
            factories_by_step=self._steps,
            output_steps=self._outputs,
            max_size=0,
            graph=self._graph,
            logger=self._logger
        )
        task_executor = executor.TaskExecutor(self._instructions)
        await task_executor.run_async(result_store, self._config_by_step, self._logger)
        if __return_values is not None:
            return {step: result_store.retrieve(step, self._steps[step])
                    for step in __return_values}
