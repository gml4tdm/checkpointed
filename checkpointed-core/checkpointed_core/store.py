from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import typing

from . import data_format
from .checkpoints import CheckpointGraph
from .handle import PipelineStepHandle
from .step import PipelineStep


class ResultStore:

    def __init__(self, *,
                 graph: CheckpointGraph,
                 output_directory: str | None,
                 checkpoint_directory: str,
                 file_by_step: dict[PipelineStepHandle, str],
                 factories_by_step: dict[PipelineStepHandle, type[PipelineStep]],
                 output_steps: frozenset[PipelineStepHandle],
                 logger: logging.Logger | None = None):
        if not data_format.is_initialised():
            data_format.initialise_format_registry()
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
            candidates = graph.get_caching_candidates(old_graph)
            valid_checkpoints = set()
            for new, old in candidates.items():
                if not self.have_checkpoint_for(old):
                    continue
                factory = factories_by_step[new]
                instance = factory(graph.config_by_step[new], self._logger)
                if instance.checkpoint_is_valid(self.retrieve_metadata(old)):
                    valid_checkpoints.add(old)
            cacheable = graph.update_caching_candidates(candidates, valid_checkpoints)
            self._remap_checkpoints(cacheable)
        with open(self._graph_file, 'wb') as f:
            pickle.dump(graph, f)

    def sub_storage(self,
                    parent_handle: PipelineStepHandle, *,
                    graph: CheckpointGraph,
                    factories_by_step: dict[PipelineStepHandle, type[PipelineStep]]) -> ResultStore:
        nested_checkpoint_directory = os.path.join(
            self._get_filename(parent_handle), 'nested'
        )
        return ResultStore(
            graph=graph,
            output_directory=None,
            checkpoint_directory=nested_checkpoint_directory,
            file_by_step={},
            factories_by_step=factories_by_step,
            output_steps=frozenset(),
            logger=self._logger
        )

    def _make_directories(self):
        os.makedirs(self._checkpoint_directory, exist_ok=True)
        os.makedirs(self._checkpoint_metadata_directory, exist_ok=True)
        os.makedirs(self._checkpoint_data_directory, exist_ok=True)
        os.makedirs(self._output_directory, exist_ok=True)

    def _remap_checkpoints(self, mapping: dict[PipelineStepHandle, PipelineStepHandle]):
        filename_mapping = {
            self._get_filename(new): self._get_filename(old)
            for new, old in mapping.items()
        } | {
            self._get_metadata_filename(new): self._get_metadata_filename(old)
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
        for file in self._get_metadata_files():
            if file not in keep:
                os.remove(file)
        for file in self._get_checkpoint_files():
            if file not in keep:
                shutil.rmtree(file)

    def store(self,
              handle: PipelineStepHandle,
              factory: type[PipelineStep],
              value: typing.Any,
              metadata: typing.Any) -> None:
        formatter = data_format.get_format(factory.get_data_format())
        if handle in self._output_steps:
            # Store result
            filename = self._get_filename(handle, is_output=True)
            if os.path.exists(filename):
                shutil.rmtree(filename)
            os.makedirs(filename)
            formatter.store(filename, value)
        # Store checkpoint
        filename = self._get_filename(handle)
        os.makedirs(filename)
        formatter.store(filename, value)
        # Store metadata
        with open(self._get_metadata_filename(handle), 'w') as file:
            json.dump(metadata, file)

    def retrieve(self,
                 handle: PipelineStepHandle,
                 factory: type[PipelineStep]) -> typing.Any:
        filename = self._get_filename(handle)
        formatter = data_format.get_format(factory.get_data_format())
        return formatter.load(filename)

    def retrieve_metadata(self, handle: PipelineStepHandle):
        with open(self._get_metadata_filename(handle), 'r') as file:
            return json.load(file)

    def have_checkpoint_for(self, handle: PipelineStepHandle) -> bool:
        return (
            os.path.exists(self._get_filename(handle)) and
            os.path.exists(self._get_metadata_filename(handle))
        )

    def get_checkpoint_filename_for(self, handle: PipelineStepHandle) -> str:
        return self._get_filename(handle)

    def _get_filename(self,
                      handle: PipelineStepHandle,
                      *, is_output=False) -> str:
        if is_output:
            if self._output_directory is None:
                raise ValueError(
                    "Output directory is not set "
                    "(trying to save the result of a sub-pipeline as output?)"
                )
            return os.path.join(
                self._output_directory,
                self._file_by_step[handle]
            )
        else:
            return os.path.join(
                self._checkpoint_directory,
                'data',
                str(handle.get_raw_identifier())
            )

    def _get_metadata_filename(self, handle: PipelineStepHandle):
        return os.path.join(
            self._checkpoint_metadata_directory,
            str(handle.get_raw_identifier()) + '.json'
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
