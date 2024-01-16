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
        self._graph = graph
        self._config_by_step = config_by_step
        self._checkpoint_graph = checkpointing.CheckpointGraph(graph, config_by_step)
        if os.path.exists(self._graph_file):
            with open(self._graph_file, 'rb') as f:
                old_graph = pickle.load(f)
            self._caching_mapping = self._checkpoint_graph.compute_checkpoint_mapping(
                old_graph, logger
            )
        else:
            self._caching_mapping = {}
        self._remap_checkpoints()
        self._valid_static_checkpoints = self._check_static_checkpoints()
        self._caching_mapping = self._checkpoint_graph.update_checkpoint_mapping(
            self._caching_mapping, self._valid_static_checkpoints, logger
        )
        self._delete_invalidated_checkpoints()
        self._dynamic_checkpoints = dict.fromkeys(
            self._checkpoint_graph.extract_dynamic_steps(self._caching_mapping), False
        )
        self._dynamic_checkpoint_requirements = self._checkpoint_graph.extract_dynamic_requirements(
            self._caching_mapping, self._logger
        )
        self._valid_dynamic_endpoints = set()
        with open(self._graph_file, 'wb') as f:
            pickle.dump(self._checkpoint_graph, f)

    def _check_static_checkpoints(self) -> set[PipelineStepHandle]:
        valid_checkpoints = set()
        factories_by_handle = {}
        for vertex in self._graph.vertices:
            factories_by_handle[vertex.handle] = vertex.factory
        for handle in self._caching_mapping:
            if not self.have_checkpoint_for(handle):
                continue
            instance = factories_by_handle[handle](self._config_by_step[handle],
                                                   self._logger)
            if not instance.checkpoint_is_valid(self.retrieve_metadata(handle)):
                continue
            valid_checkpoints.add(handle)
        return valid_checkpoints

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
        to_rename = []
        for new, old in self._caching_mapping.items():
            if not os.path.exists(self._get_metadata_filename(old)):
                continue
            os.rename(
                self._get_checkpoint_filename(old),
                self._get_checkpoint_filename(new) + '_temp'
            )
            os.rename(
                self._get_metadata_filename(old),
                self._get_metadata_filename(new) + '_temp'
            )
            to_rename.append(new)
        for new in to_rename:
            os.rename(
                self._get_checkpoint_filename(new) + '_temp',
                self._get_checkpoint_filename(new)
            )
            os.rename(
                self._get_metadata_filename(new) + '_temp',
                self._get_metadata_filename(new)
            )

    def _delete_old_checkpoints(self):
        keep_files = {
            self._get_checkpoint_filename(h) for h in self._caching_mapping.values()
        } | {
            self._get_metadata_filename(h) for h in self._caching_mapping.values()
        }
        self._delete_checkpoints(keep_files)

    def _delete_invalidated_checkpoints(self):
        keep_files = {
            self._get_checkpoint_filename(h) for h in self._caching_mapping
        } | {
            self._get_metadata_filename(h) for h in self._caching_mapping
        }
        self._delete_checkpoints(keep_files)

    def _delete_checkpoints(self, keep: set[str]):
        for file in self._get_metadata_files():
            if file not in keep:
                os.remove(file)
        for file in self._get_checkpoint_files():
            if file not in keep:
                shutil.rmtree(file)

    def mark_checkpoint(self, handle: PipelineStepHandle, tainted: bool):
        if handle not in self._dynamic_checkpoints:
            raise ValueError(f'Cannot mark checkpoint state of non-dynamic checkpoint: {handle}')
        elif self._dynamic_checkpoints[handle]:
            raise ValueError(f'Checkpoint {handle} already marked')
        self._dynamic_checkpoints[handle] = True
        if not tainted:
            return
        self._valid_dynamic_endpoints.add(handle)
        self._caching_mapping = self._checkpoint_graph.update_checkpoint_mapping(
            self._caching_mapping,
            self._valid_static_checkpoints | self._valid_dynamic_endpoints,
            self._logger
        )
        self._delete_invalidated_checkpoints()

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
        for requirement in self._dynamic_checkpoint_requirements.get(handle, set()):
            if not self._dynamic_checkpoints[requirement]:
                raise RuntimeError(
                    f'Attempting to retrieve checkpoint data for step {handle} '
                    f'before taint value of dynamic checkpoint '
                    f'{requirement} has been set'
                )
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
