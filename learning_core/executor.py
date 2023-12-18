import asyncio

from .handle import PipelineStepHandle
from .instructions import Instruction, Start, Sync


class TaskExecutor:

    def __init__(self, instructions: list[Instruction]):
        self._pending: list[Start] = []
        self._blocked: list[Sync] = []
        for instruction in instructions:
            if isinstance(instruction, Start):
                self._pending.append(instruction)
            elif isinstance(instruction, Sync):
                self._blocked.append(instruction)
            else:
                raise NotImplementedError(f"Instruction {instruction} is not supported")
        self._active = []
        self._done = set()

    def loop(self, config_by_step: dict[PipelineStepHandle, dict]):
        loop = asyncio.get_running_loop()

        while self._pending or self._blocked or self._active:
            done, self._active = asyncio.wait(self._active, return_when=asyncio.FIRST_COMPLETED)
            self._unblock_tasks()
            for task in self._pending:
                instance = task.factory(config_by_step[task.handle])
                async def wrap():
                    result = await instance.execute()
                    return result
    def _unblock_tasks(self):
        for task in self._blocked.copy():
            if task.steps <= self._done:
                self._blocked.remove(task)
                self._pending.extend(task.then)



