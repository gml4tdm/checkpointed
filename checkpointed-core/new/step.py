from __future__ import annotations

import abc
import logging
import typing

from .parameters import ArgumentConsumer, Config


class PipelineStep(ArgumentConsumer, abc.ABC):
    """Abstract definition for a single pipeline step.

    Pipeline steps are always instantiated with a configuration
    and logger object.

    During runtime, the task executor will set various other
    runtime-only properties, such as the execution context,
    input storage formats, and streamed inputs.

    A number of abstract methods should be defined for a
    pipeline step. First of all, the valid incoming connections
    should be defined. This is done defining the method
    `supported_inputs(cls) -> dict[str | type(...), tuple[type]]`:

        This method should return a dict, the keys of which
        define the names of the valid incoming connections
        for the step.For instance, a machine learning step may
        have two arguments 'features' and 'labels',
        which would be the keys of the dictionary.

        The value for each key should be a tuple of types.
        The library will make sure that when input labelled
        `X` is passed, it satisfies
        `issubclass(inputs[X], cls.supported_inputs()[X])`.

        Pipeline steps which accept arbitrary arguments should
        include Ellipsis (`...`) as a key in the dictionary.
        Note, however, that such steps can still return the
        names of other mandatory incoming connections.

    TODO: finish documentation
    """

    def __init__(self, config: dict[str, typing.Any], logger: logging.Logger):
        self.config = self.parse_arguments(config)
        self.logger = logger
        self._input_storage_formats: dict[str, str] | None = None
        self._execution_context: Config | None = None
        self._streamed_inputs: set[str] | None = None

    # ========== Getters and Setters for External Metadata ==========

    @property
    def execution_context(self) -> Config:
        if self._execution_context is None:
            raise ValueError("Execution context not set")
        return self._execution_context

    @execution_context.setter
    def execution_context(self, context: Config):
        if self._execution_context is not None:
            raise ValueError("Execution context already set")
        self._execution_context = context

    @property
    def input_storage_formats(self) -> dict[str, str]:
        if self._input_storage_formats is None:
            raise ValueError("Input storage formats not set")
        return self._input_storage_formats

    @input_storage_formats.setter
    def input_storage_formats(self, formats: dict[str, str]):
        if self._input_storage_formats is not None:
            raise ValueError("Input storage formats already set")
        self._input_storage_formats = formats

    @property
    def streamed_inputs(self) -> set[str]:
        if self._streamed_inputs is None:
            raise ValueError("Streamed inputs not set")
        return self._streamed_inputs

    @streamed_inputs.setter
    def streamed_inputs(self, inputs: set[str]):
        if self._streamed_inputs is not None:
            raise ValueError("Streamed inputs already set")
        self._streamed_inputs = inputs

    # ========== Input Step Definitions ==========

    @classmethod
    @abc.abstractmethod
    def supported_inputs(cls) -> dict[str | type(...), tuple[type]]:
        pass

    @classmethod
    @abc.abstractmethod
    def supported_streamed_inputs(cls) -> dict[str | type(...), tuple[type]]:
        pass

    # ========== Main Implementation ==========

    @abc.abstractmethod
    async def execute(self, *,
                      streamed_inputs: list[str] | None = None,
                      **inputs) -> typing.Any:
        pass

    # ========== Storage and Checkpointing Functions ==========

    @classmethod
    @abc.abstractmethod
    def get_output_storage_format(cls) -> str:
        pass

    @abc.abstractmethod
    def get_checkpoint_metadata(self) -> typing.Any:
        pass

    @abc.abstractmethod
    def checkpoint_is_valid(self, metadata: typing.Any) -> bool:
        pass

    @classmethod
    def has_dynamic_checkpoint(cls) -> bool:
        """Special method which is used to enable checkpointing of
        nested or scatter/gather steps.

        If this method returns `True`, then the `execute` method
        will be called regardless of the checkpoint status.
        However, after the step has been executed, the
        `dynamic_checkpoint_is_valid` method is called to
        retroactively determine validity of the checkpoint,
        and to thus determine checkpoint invalidation for all
        following steps.
        """
        return False

    def dynamic_checkpoint_is_valid(self) -> bool:
        raise NotImplementedError(
            f'Step {self.__class__.__name__} with dynamic checkpointing '
            f'behaviour does not implement `dynamic_checkpoint_is_valid`.'
        )
