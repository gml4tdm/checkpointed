import json
import logging
import os
import pickle
import typing

from .checkpoints import CheckpointGraph
from .handle import PipelineStepHandle
from .step import PipelineStep


class ResultStore:

    def __init__(self, *,
                 graph: CheckpointGraph,
                 output_directory: str,
                 checkpoint_directory: str,
                 file_by_step: dict[PipelineStepHandle, str],
                 output_steps: frozenset[PipelineStepHandle],
                 max_size: int,
                 logger: logging.Logger | None = None):
        self._output_directory = output_directory
        self._checkpoint_directory = checkpoint_directory
        self._checkpoint_metadata_directory = os.path.join(
            self._checkpoint_directory, 'metadata'
        )
        self._checkpoint_data_directory = os.path.join(
            self._checkpoint_directory, 'data'
        )
        self._file_by_step = file_by_step
        self._output_steps = output_steps
        self._max_size = max_size
        if logger is None:
            self._logger = logging.getLogger()
        else:
            self._logger = logger
        self._make_directories()
        self._graph_file = os.path.join(
            self._checkpoint_metadata_directory,
            'graph.pickle'
        )
        if os.path.exists(self._graph_file):
            with open(self._graph_file, 'rb') as f:
                old_graph = pickle.load(f)
            self._remap_checkpoints(
                graph.get_largest_isomorphic_prefix(old_graph)
            )
        with open(self._graph_file, 'wb') as f:
            pickle.dump(graph, f)

    def _make_directories(self):
        os.makedirs(self._checkpoint_directory, exist_ok=True)
        os.makedirs(self._checkpoint_metadata_directory, exist_ok=True)
        os.makedirs(self._checkpoint_data_directory, exist_ok=True)
        os.makedirs(self._output_directory, exist_ok=True)

    def _remap_checkpoints(self, mapping: dict[PipelineStepHandle, PipelineStepHandle]):
        filename_mapping = {
            self._get_filename(new): self._get_filename(new)
            for new, old in mapping.items()
        }
        self._delete_old_checkpoints(keep=set(filename_mapping.values()))
        # Rename files in two-step to prevent overwriting files
        to_rename = []
        for new, old in filename_mapping.items():
            if not os.path.exists(old):
                continue
            os.rename(old, new + '_temp')
            to_rename.append(new)
        for new in to_rename:
            os.rename(new + '_temp', new)

    def _delete_old_checkpoints(self, keep: set[str]):
        for file_list in [self._get_metadata_files(), self._get_checkpoint_files()]:
            for file in file_list:
                path, filename = os.path.split(file)
                if filename not in keep and os.path.exists(filename):
                    os.remove(file)

    def store(self,
              handle: PipelineStepHandle,
              factory: type[PipelineStep],
              value: typing.Any,
              metadata: typing.Any) -> None:
        if handle in self._output_steps:
            filename = self._get_filename(handle, is_output=True)
            factory.save_result(filename, value)
        filename = self._get_filename(handle)
        factory.save_result(filename, value)
        with open(self._get_metadata_filename(handle), 'w') as file:
            json.dump(metadata, file)

    def retrieve(self,
                 handle: PipelineStepHandle,
                 factory: type[PipelineStep]) -> typing.Any:
        filename = self._get_filename(handle)
        return factory.load_result(filename)

    def have_checkpoint_for(self, handle: PipelineStepHandle) -> bool:
        return os.path.exists(self._get_filename(handle))

    def _get_filename(self,
                      handle: PipelineStepHandle,
                      *, is_output=False) -> str:
        if is_output:
            return os.path.join(
                self._output_directory,
                self._file_by_step[handle]
            )
        else:
            return os.path.join(
                self._checkpoint_directory,
                str(handle.get_raw_identifier())
            )

    def _get_metadata_filename(self, handle: PipelineStepHandle):
        return os.path.join(
            self._checkpoint_metadata_directory,
            str(handle.get_raw_identifier())
        )

    def _get_metadata_files(self) -> list[str]:
        return [
            os.path.join(self._checkpoint_metadata_directory, filename)
            for filename in os.listdir(self._checkpoint_metadata_directory)
        ]

    def _get_checkpoint_files(self) -> list[str]:
        return [
            os.path.join(self._checkpoint_data_directory, filename)
            for filename in os.listdir(self._checkpoint_data_directory)
        ]

    def _get_output_files(self) -> list[str]:
        return [
            os.path.join(self._output_directory, filename)
            for filename in os.listdir(self._output_directory)
        ]
