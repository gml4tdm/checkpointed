from __future__ import annotations

import abc
import graphlib
import logging
import typing

from . import arguments
from .errors import ArgumentParsingError
from .core import ConfigFactory, NoSuchSetting, NotSet


class _NullLogger:
    def debug(self, *args, **kwargs): pass

    def info(self, *args, **kwargs): pass

    def warning(self, *args, **kwargs): pass

    def warn(self, *args, **kwargs): pass

    def error(self, *args, **kwargs): pass

    def critical(self, *args, **kwargs): pass


class ArgumentConsumer:

    @classmethod
    @abc.abstractmethod
    def get_arguments(cls) -> dict[str, arguments.Argument]:
        return {}

    @classmethod
    @abc.abstractmethod
    def get_constraints(cls) -> list[arguments.Constraint]:
        return []

    def validate_arguments(self,
                           params: dict[str, typing.Any],
                           logger: logging.Logger | None = None) -> dict[str, typing.Any]:
        if logger is None:
            logger = _NullLogger()
        self._check_arg_dependencies()
        parsed = self._parse_args(params, logger)
        try:
            self._impose_constraints(parsed)
        except ArgumentParsingError as e:
            logger.error(f'Constraint failure: {e}')
            raise e
        return parsed

    def _check_arg_dependencies(self):
        graph = {}
        for name, arg in self.get_arguments().items():
            graph[name] = set(arg.depends_on())
        sorter = graphlib.TopologicalSorter(graph)
        try:
            return list(sorter.static_order())
        except graphlib.CycleError as e:
            msg = 'Cannot parse arguments because of cycle in enabling conditions'
            raise Exception(msg) from e

    def _parse_args(self,
                    params: dict[str, typing.Any],
                    logger: logging.Logger) -> dict[str, typing.Any]:
        logger.info(f'Parsing argument list for {self.__class__.__name__!r}')
        result = {}
        for name, arg in self.get_arguments().items():
            logger.info(f'Parsing argument {name!r}')
            try:
                if not arg.is_enabled(ConfigFactory.dict_config(result)):
                    logger.info(f'Skipping disabled argument: {name}')
                    continue
            except (NoSuchSetting, NotSet) as e:
                raise ValueError(
                    f'Error while evaluating enablement constraint. '
                    f'Is one of the required arguments not enabled?') from e
            if name in params:
                try:
                    result[name] = arg.validate(params[name])
                except ArgumentParsingError as e:
                    logger.error(f'Error while parsing argument {name}: {e}')
                    raise e
            elif arg.has_default:
                logger.info(f'Applying default for argument {name!r}')
                result[name] = arg.default
            else:
                logger.error(f'Missing required argument {name!r}')
                raise ArgumentParsingError(r'Missing required argument {name!r}')
        if extra := params.keys() - result.keys():
            formatted = ', '.join(sorted(extra))
            logger.info(f'Got unknown arguments: {formatted}')
            raise ArgumentParsingError(f'Got unknown arguments: {formatted}')
        return result

    def _impose_constraints(self, parsed: dict[str, typing.Any]):
        constraints: list[arguments.Constraint] = self.get_constraints()
        conf = ConfigFactory.dict_config(parsed)
        for constraint in constraints:
            if not constraint.impose(conf):
                raise ArgumentParsingError(f'Constraint failure: {constraint.description()}')
