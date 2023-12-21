import os
import typing

from checkpointed.handle import PipelineStepHandle
from checkpointed.step import PipelineStep


class ResultStore:

    def __init__(self, *,
                 output_directory: str,
                 checkpoint_directory: str,
                 file_by_step: dict[PipelineStepHandle, str],
                 output_steps: frozenset[PipelineStepHandle],
                 max_size: int):
        self._output_directory = output_directory
        self._checkpoint_directory = checkpoint_directory
        self._file_by_step = file_by_step
        self._output_steps = output_steps
        self._max_size = max_size

    def store(self,
              handle: PipelineStepHandle,
              factory: type[PipelineStep],
              value: typing.Any) -> None:
        if handle in self._output_steps:
            os.makedirs(self._output_directory, exist_ok=True)
            filename = self._get_filename(handle, is_output=True)
            factory.save_result(filename, value)
        os.makedirs(self._checkpoint_directory, exist_ok=True)
        filename = self._get_filename(handle)
        factory.save_result(filename, value)

    def retrieve(self,
                 handle: PipelineStepHandle,
                 factory: type[PipelineStep]) -> typing.Any:
        filename = self._get_filename(handle)
        return factory.load_result(filename)

    def _get_filename(self,
                      handle: PipelineStepHandle,
                      *, is_output=False) -> str:
        if is_output:
            return os.path.join(
                self._output_directory,
                'output',
                self._file_by_step[handle]
            )
        else:
            return os.path.join(
                self._checkpoint_directory,
                'checkpoints',
                str(handle)
            )
