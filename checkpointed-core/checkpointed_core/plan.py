import asyncio
import logging
import typing

from .data_store import ResultStore
from .graph import PipelineGraph
from .handle import PipelineStepHandle
from .instructions import Instruction
from .executor import TaskExecutor


class ExecutionPlan:

    def __init__(self, *,
                 name: str,
                 instructions: list[Instruction],
                 graph: PipelineGraph,
                 config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]]):
        self.name = name
        self._instructions = instructions
        self._graph = graph
        self._config_by_step = config_by_step

    def execute(self, *,
                output_directory='',
                checkpoint_directory='',
                logger: logging.Logger | None = None,
                _precomputed_inputs: dict[PipelineStepHandle, typing.Any] | None = None,
                _return_values: set[PipelineStepHandle] | None = None,
                loop: asyncio.AbstractEventLoop | None = None):
        if loop is None:
            loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.execute_async(
                output_directory=output_directory,
                checkpoint_directory=checkpoint_directory,
                logger=logger,
                _precomputed_inputs=_precomputed_inputs,
                _return_values=_return_values,
                loop=loop
            )
        )

    async def execute_async(self, *,
                            output_directory='',
                            checkpoint_directory='',
                            logger: logging.Logger | None = None,
                            _precomputed_inputs: dict[PipelineStepHandle, typing.Any] | None = None,
                            _return_values: set[PipelineStepHandle] | None = None,
                            loop: asyncio.AbstractEventLoop):
        if logger is None:
            logger = logging.getLogger(__name__)
        if _precomputed_inputs is None:
            _precomputed_inputs = {}
        if _return_values is None:
            _return_values = set()
        result_store = ResultStore(
            graph=self._graph,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            config_by_step=self._config_by_step,
            logger=logger,
        )
        executor = TaskExecutor(loop)
        await executor.run_session(
            instructions=self._instructions,
            result_store=result_store,
            config_by_step=self._config_by_step,
            preloaded_inputs_by_step=_precomputed_inputs,
            logger=logger
        )
        if _return_values is not None:
            steps = {
                node.handle: node.factory for node in self._graph.vertices
            }
            return {step: result_store.retrieve(step, steps[step])
                    for step in _return_values}
