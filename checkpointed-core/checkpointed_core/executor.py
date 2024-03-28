import asyncio
import logging
import traceback
import typing

from .parameters import ConfigFactory
from .handle import PipelineStepHandle
from .instructions import Instruction, Start, Sync
from .data_store import ResultStore
from .step import PipelineStep


class TaskExecutor:

    def __init__(self, loop=None):
        self._loop = loop if loop is not None else asyncio.get_event_loop()

    @property
    def loop(self):
        return self._loop

    async def run_session(self, *,
                          instructions: list[Instruction],
                          result_store: ResultStore,
                          config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                          preloaded_inputs_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                          logger: logging.Logger):
        session = Session(
            self._loop,
            self,
            instructions=instructions,
            result_store=result_store,
            config_by_step=config_by_step,
            preloaded_inputs_by_step=preloaded_inputs_by_step,
            logger=logger
        )
        await session.run()


class Session:

    def __init__(self,
                 loop,
                 executor: TaskExecutor, *,
                 instructions: list[Instruction],
                 result_store: ResultStore,
                 config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                 preloaded_inputs_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                 logger: logging.Logger):
        self._loop = loop
        self._executor = executor
        self._result_store = result_store
        self._config_by_step = config_by_step
        self._preloaded_inputs_by_step = preloaded_inputs_by_step
        self._logger = logger
        self._pending: list[Start] = []
        self._blocked: list[Sync] = []
        for instruction in instructions:
            if isinstance(instruction, Start):
                self._pending.append(instruction)
            elif isinstance(instruction, Sync):
                self._blocked.append(instruction)
            else:
                raise NotImplementedError(f"Instruction {instruction} is not supported")
        self._active = set()
        self._done = set()

    @staticmethod
    def _get_config_factory() -> ConfigFactory:
        config_factory = ConfigFactory()
        config_factory.register_namespace('system')
        config_factory.register_namespace('system.step')
        config_factory.register_namespace('system.step.storage')
        # config_factory.register('system.step.is-output-step')
        config_factory.register('system.step.storage.current-checkpoint-directory')
        config_factory.register('system.step.handle')
        config_factory.register_namespace('system.executor')
        # config_factory.register('system.executor.resource-manager')    # maybe in the future
        config_factory.register('system.executor.storage-manager')
        config_factory.register('system.executor.current-executor')
        return config_factory

    async def run(self):
        while self._pending or self._blocked or self._active:
            self._unblock_pending_tasks()
            self._start_pending_tasks()
            await self._handle_done_tasks()

    def _unblock_pending_tasks(self):
        for task in self._blocked.copy():
            if task.steps <= self._done:
                self._logger.info(f"Unblocking {len(task.then)} tasks")
                self._blocked.remove(task)
                self._pending.extend(task.then)

    async def _handle_done_tasks(self):
        done, self._active = await asyncio.wait(
            self._active, return_when=asyncio.FIRST_COMPLETED
        )
        exceptions = []
        for data in done:
            if (exc := data.exception()) is not None:
                self._emit_task_error(exc, do_raise=False)
                exceptions.append(exc)
            handle, factory = data.result()
            self._logger.info(f'Task {handle} finished')
            self._done.add(handle)

    def _emit_task_error(self, exc: BaseException, *, do_raise=True):
        self._logger.error(f'Error in task: {exc}')
        tb = ''.join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self._logger.error(f'Traceback:\n\n{tb}')
        if do_raise:
            raise exc

    def _start_pending_tasks(self):
        while self._pending:
            task = self._pending.pop()
            self._logger.info(f"Starting pending task {task.step}")
            self._active.add(asyncio.Task(self._build_task_wrapper(task), loop=self._loop))

    def _prepare_task_inputs(self,
                             task_handle: PipelineStepHandle,
                             inputs: list[tuple[PipelineStepHandle, type[PipelineStep], str]],
                             logger: logging.Logger):
        args = {}
        input_formats = {}
        for handle, factory, name in inputs:
            if name not in self._preloaded_inputs_by_step.get(handle, {}):
                logger.info(f'Loading input {name} ({handle}, type {factory.__name__}) '
                            f'for task {task_handle}')
                args[name] = self._result_store.retrieve(handle, factory)
                input_formats[name] = factory.get_output_storage_format()
            else:
                logger.info(f'Loading input {name} ({handle}, type {factory.__name__}) '
                            f'for task {task_handle} from preloaded inputs')
                args[name] = self._preloaded_inputs_by_step[task_handle][name]
                input_formats[name] = factory.get_output_storage_format()
        # Special case for scatter gather inputs
        config = self._get_config_factory().build_config('system')
        config.set('system.step.handle', task_handle)
        config.set(
            'system.step.storage.current-checkpoint-directory',
            self._result_store.get_checkpoint_filename_for(task_handle)
        )
        config.set('system.executor.storage-manager', self._result_store)
        config.set('system.executor.current-executor', self._executor)
        return args, input_formats, config

    def _build_task_wrapper(self, task: Start):
        async def wrapper(_handle=task.step,
                          _factory=task.factory,
                          _inputs=tuple(task.inputs)):
            logger = self._logger.getChild(str(_handle))
            logger.info('Checking checkpoint...')
            if self._can_skip(_handle, _factory):
                logger.info(f'Skipping task (found valid checkpoint)')
                return _handle, _factory
            logger.info(f'Running task (no valid checkpoint)')
            instance = _factory(self._config_by_step[_handle], logger)
            inputs, input_formats, config = self._prepare_task_inputs(
                _handle, _inputs, logger
            )
            instance.input_storage_formats = input_formats
            instance.execution_context = config
            result = await instance.execute(**dict(inputs))
            logger.info(f'Storing result')
            self._result_store.store(_handle,
                                     _factory,
                                     result,
                                     instance.get_checkpoint_metadata())
            if _factory.has_dynamic_checkpoint():
                self._result_store.mark_checkpoint(
                    _handle, instance.dynamic_checkpoint_is_valid()
                )
            logger.info(f'Finished task')
            return _handle, _factory

        return wrapper()

    def _can_skip(self, handle: PipelineStepHandle, factory: type[PipelineStep]) -> bool:
        if not self._result_store.have_checkpoint_for(handle):
            return False
        instance = factory(self._config_by_step[handle], self._logger)
        metadata = self._result_store.retrieve_metadata(handle)
        return instance.checkpoint_is_valid(metadata)
