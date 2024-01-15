from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import typing

from . import checkpointing
from . import data_format
from .graph import PipelineGraph
from .handle import PipelineStepHandle
from .step import PipelineStep


class ResultStore:

    def __init__(self, *,
                 output_directory: str | None,
                 checkpoint_directory: str,
                 graph: PipelineGraph,
                 config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]],
                 logger: logging.Logger):
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
        self._output_file_by_step = {
            node.handle: node.output_filename
            for node in graph.vertices
            if node.is_output
        }
        self._logger = logger
        self._make_directories()
        # Load checkpointing
        self._graph_file = os.path.join(
            self._checkpoint_metadata_directory,
            'graph.pickle'
        )
        self._graph = checkpointing.CheckpointGraph(graph, config_by_step)
        if os.path.exists(self._graph_file):
            with open(self._graph_file, 'rb') as f:
                old_graph = pickle.load(f)
            self._caching_mapping = self._graph.compute_checkpoint_mapping(
                old_graph, logger
            )
            self._remap_checkpoints()
        else:
            self._caching_mapping = {}
        with open(self._graph_file, 'wb') as f:
            pickle.dump(self._graph, f)

    def sub_storage(self,
                    parent_handle: PipelineStepHandle, *,
                    graph: PipelineGraph,
                    config_by_step: dict[PipelineStepHandle, dict[str, typing.Any]]) -> ResultStore:
        nested_checkpoint_directory = os.path.join(
            self._get_checkpoint_filename(parent_handle), 'nested'
        )
        return ResultStore(
            graph=graph,
            output_directory=None,
            checkpoint_directory=nested_checkpoint_directory,
            config_by_step=config_by_step,
            logger=self._logger
        )

    def _make_directories(self):
        os.makedirs(self._checkpoint_directory, exist_ok=True)
        os.makedirs(self._checkpoint_metadata_directory, exist_ok=True)
        os.makedirs(self._checkpoint_data_directory, exist_ok=True)
        if self._output_directory is not None:
            os.makedirs(self._output_directory, exist_ok=True)

    def _remap_checkpoints(self):
        keep_files = {
            self._get_checkpoint_filename(h) for h in self._caching_mapping.values()
        } | {
            self._get_metadata_filename(h) for h in self._caching_mapping.values()
        }
        self._delete_old_checkpoints(keep_files)
        for new, old in self._caching_mapping.items():
            os.rename(
                self._get_checkpoint_filename(old),
                self._get_checkpoint_filename(new) + '_temp'
            )
            os.rename(
                self._get_metadata_filename(old),
                self._get_metadata_filename(new) + '_temp'
            )
        for new in self._caching_mapping:
            os.rename(
                self._get_checkpoint_filename(new) + '_temp',
                self._get_checkpoint_filename(new)
            )
            os.rename(
                self._get_metadata_filename(new) + '_temp',
                self._get_metadata_filename(new)
            )

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
        formatter = data_format.get_format(factory.get_output_storage_format())
        if handle in self._output_file_by_step:
            # Store result
            filename = self._get_output_filename(handle)
            if os.path.exists(filename):
                shutil.rmtree(filename)
            os.makedirs(filename)
            formatter.store(filename, value)
        # Store checkpoint
        filename = self._get_checkpoint_filename(handle)
        os.makedirs(filename, exist_ok=True)
        formatter.store(filename, value)
        # Store metadata
        with open(self._get_metadata_filename(handle), 'w') as file:
            json.dump(metadata, file)

    def retrieve(self,
                 handle: PipelineStepHandle,
                 factory: type[PipelineStep]) -> typing.Any:
        filename = self._get_checkpoint_filename(handle)
        formatter = data_format.get_format(factory.get_output_storage_format())
        return formatter.load(filename)

    def retrieve_metadata(self, handle: PipelineStepHandle):
        with open(self._get_metadata_filename(handle), 'r') as file:
            return json.load(file)

    def have_checkpoint_for(self, handle: PipelineStepHandle) -> bool:
        return (
            os.path.exists(self._get_checkpoint_filename(handle)) and
            os.path.exists(self._get_metadata_filename(handle))
        )

    def get_checkpoint_filename_for(self, handle: PipelineStepHandle) -> str:
        return self._get_checkpoint_filename(handle)

    def _get_checkpoint_filename(self, handle: PipelineStepHandle) -> str:
        return os.path.join(
            self._checkpoint_directory,
            'data',
            str(handle.get_raw_identifier())
        )

    def _get_output_filename(self, handle: PipelineStepHandle) -> str:
        if self._output_directory is None:
            raise ValueError('No output directory is set. '
                             'Trying to save the output of a sub-pipeline?')
        return os.path.join(
            self._output_directory,
            self._output_file_by_step[handle]
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
