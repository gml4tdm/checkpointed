from __future__ import annotations

import abc
import graphlib
import logging
import re
import typing

from . import core
from .errors import ArgumentParsingError
from . import schemas
from . import constraints


class Argument(abc.ABC):

    _NOT_SET = object()

    def __init__(self,
                 name: str,
                 description: str,
                 data_type,
                 default=_NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        self._name = name
        self._description = description
        self._data_type = data_type
        self._default = default
        self.enabled_if = enabled_if

    @property
    def argument_name(self):
        return self._name

    @property
    def argument_description(self):
        return self._description

    @property
    def default(self):
        return self._default

    @property
    def has_default(self):
        return self._default is not self._NOT_SET

    @property
    def argument_type(self):
        return self._data_type

    def depends_on(self) -> list[str]:
        if self.enabled_if is None:
            return []
        return self.enabled_if.involved_arguments

    def is_enabled(self, conf: core.Config) -> bool:
        return self.enabled_if is None or self.enabled_if.evaluate(conf)

    @abc.abstractmethod
    def validate(self, value):
        pass

    @abc.abstractmethod
    def get_json_spec(self):
        pattern = re.compile(r'([A-Z])')
        parts = [
            x
            for x in pattern.split(self.__class__.__name__.replace('Argument', ''))
            if x
        ]
        arg_type = '-'.join(
            f'{x}{y}' for x, y in zip(parts[::2], parts[1::2])
        ).lower()
        return {
            "name": self._name,
            "description": self._description,
            "type": self._data_type.__name__,
            'argument_type': arg_type,
            "has-default": self.has_default,
            "default": self._default if self.has_default else None,
            'enabled-if': self.enabled_if.to_json() if self.enabled_if is not None else None
        }

    def raise_invalid(self, msg):
        raise ArgumentParsingError(f"Argument {self.argument_name!r} is invalid: {msg}")


class ListArgument(Argument):

    def __init__(self, *,
                 default=Argument._NOT_SET,
                 inner: Argument):
        super().__init__(inner.argument_name,
                         f'{inner.argument_description} (multi-valued)',
                         list,
                         default,
                         enabled_if=inner.enabled_if)
        self._inner = inner

    def validate(self, value):
        if not isinstance(value, list):
            self.raise_invalid(f'Expected a list of values')
        return [self._inner.validate(x) for x in value]

    def get_json_spec(self):
        return super().get_json_spec() | {
            'inner': self._inner.get_json_spec()
        }


class _NumericalArgument(Argument, abc.ABC):

    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET,
                 minimum: float | None = None,
                 maximum: float | None = None, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, self._get_data_type(), default, enabled_if=enabled_if)
        self._min = minimum
        self._max = maximum

    @staticmethod
    @abc.abstractmethod
    def _get_data_type():
        pass

    def validate(self, value):
        if not isinstance(value, self._get_data_type()):
            self.raise_invalid(f"Must be {self._get_data_type().__name__}, "
                               f"got {value.__class__.__name__}")
        if self._min is not None and value < self._min:
            self.raise_invalid(f"Must be >= {self._min}")
        if self._max is not None and value > self._max:
            self.raise_invalid(f"Must be <= {self._max}")
        return value

    def get_json_spec(self):
        return super().get_json_spec() | {"minimum": self._min, "maximum": self._max}


class FloatArgument(_NumericalArgument):

    @staticmethod
    def _get_data_type():
        return float


class IntArgument(_NumericalArgument):

    @staticmethod
    def _get_data_type():
        return int


class EnumArgument(Argument):

    def __init__(
        self,
        name: str,
        description: str,
        default=Argument._NOT_SET,
        options: list[str] = None, *,
        enabled_if: constraints.ValueExpression | None = None
    ):
        super().__init__(name, description, str, default, enabled_if=enabled_if)
        if options is None:
            self._options = []
        else:
            self._options = options

    def validate(self, value):
        if not isinstance(value, str):
            self.raise_invalid(f"Must be string, got {value.__class__.__name__}")
        if value not in self._options:
            self.raise_invalid(f'Must be one of {", ".join(self._options)} (got {value})')
        return value

    def get_json_spec(self):
        return super().get_json_spec() | {"options": self._options}


class DynamicEnumArgument(EnumArgument):

    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 lookup_map,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, default, list(lookup_map), enabled_if=enabled_if)


class BoolArgument(Argument):
    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, bool, default, enabled_if=enabled_if)

    def validate(self, value):
        if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
            return value
        self.raise_invalid(f"Must be Boolean, got {value.__class__.__name__}")

    def get_json_spec(self):
        return super().get_json_spec() | {}


class StringArgument(Argument):
    def __init__(self,
                 name: str,
                 description: str,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, str, default, enabled_if=enabled_if)

    def validate(self, value):
        if not isinstance(value, str):
            self.raise_invalid(f"Must be string, got {value.__class__.__name__}")
        return value

    def get_json_spec(self):
        return super().get_json_spec() | {}


class JSONArgument(Argument):

    def __init__(self,
                 name: str,
                 description: str,
                 schema: schemas.JSONSchema,
                 default=Argument._NOT_SET, *,
                 enabled_if: constraints.ValueExpression | None = None):
        super().__init__(name, description, object, default, enabled_if=enabled_if)
        self._schema = schema

    def validate(self, value):
        try:
            self._schema.validate(value)
        except schemas.SchemaValueMismatch as e:
            self.raise_invalid(f'Value {value} does not match schema: {e}')
        return value

    def get_json_spec(self):
        super().get_json_spec() | {
            'schema': self._schema.serialize()
        }


class NestedArgumentGroup(Argument):

    def __init__(self, *,
                 name: str,
                 description: str,
                 nested: dict[str, Argument],
                 constraint_items: list[constraints.Constraint],
                 enabled_if: constraints.ValueExpression | None = None):
        if all(v.has_default for v in nested.values()):
            default = {k: v.default for k, v in nested.items()}
        else:
            default = self._NOT_SET
        super().__init__(
            name, description, dict, default, enabled_if=enabled_if
        )
        self._nested = nested
        self._constraints = constraint_items
        self._config_factory = core.ConfigFactory()
        for name, arg in self._nested.items():
            if isinstance(arg, NestedArgumentGroup):
                self._config_factory.mount_sub_config(f'{self.argument_name}.{name}', arg._config_factory)
            else:
                self._config_factory.register(f'{self.argument_name}.{name}')

    def get_json_spec(self):
        return super().get_json_spec() | {
            'nested': {
                key: value.get_json_spec() for key, value in self._nested.items()
            },
            'constraints': [x.to_json() for x in self._constraints]
        }

    def validate(self, value):
        logger = logging.getLogger(self.__class__.__name__)
        return self._validate_arguments(value, logger)

    def validate_with_logging(self, value, logger: logging.Logger):
        return self._validate_arguments(value, logger)

    def _validate_arguments(self,
                            params: dict[str, typing.Any],
                            logger: logging.Logger | None = None) -> core.Config:
        order, graph = self._check_arg_dependencies()
        parsed = self._parse_args(params, order, graph, logger)
        try:
            self._impose_constraints(parsed)
        except ArgumentParsingError as e:
            logger.error(f'Constraint failure: {e}')
            raise e
        return parsed

    def _check_arg_dependencies(self) -> tuple[list[str], dict[str, set[str]]]:
        graph = {}
        required = set()
        for name, arg in self._nested.items():
            graph[name] = set(arg.depends_on())
            required |= set(arg.depends_on())
        sorter = graphlib.TopologicalSorter(graph)
        try:
            return list(sorter.static_order()), graph
        except graphlib.CycleError as e:
            msg = 'Cannot parse arguments because of cycle in enabling conditions.'
            raise Exception(msg) from e

    def _parse_args(self,
                    params: dict[str, typing.Any],
                    order: list[str],
                    graph: dict[str, set[str]],
                    logger: logging.Logger) -> core.Config:
        disabled = set()
        result = {}
        for name in order:
            argument = self._nested[name]
            logger.info(f'Parsing argument {name!r}')
            disabled_for_argument = disabled & graph[name]
            if disabled_for_argument:
                logger.info(
                    f'Skipping disabled argument: {name} '
                    f'(disabled because {_fmt_list(disabled_for_argument)} is/are disabled)'
                )
                disabled.add(name)
                continue
            if not self._eval_is_enabled(argument, result, logger):
                logger.info(f'Skipping disabled argument: {name}')
                disabled.add(name)
                continue
            if name in params:
                try:
                    result[name] = argument.validate(params[name])
                except ArgumentParsingError as e:
                    logger.error(f'Error while parsing argument {name}: {e}')
                    raise e
            elif argument.has_default:
                logger.info(f'Applying default for argument {name!r}')
                result[name] = argument.default
            else:
                logger.error(f'Missing required argument {name!r}')
                raise ArgumentParsingError(f'Missing required argument {name!r}')
        if extra := params.keys() - result.keys():
            formatted = ', '.join(sorted(extra))
            logger.info(f'Got unknown arguments: {formatted}')
            raise ArgumentParsingError(f'Got unknown arguments: {formatted}')
        return self._build_final_config(result)

    def _build_final_config(self, parsed: dict[str, typing.Any]) -> core.Config:
        conf = self._config_factory.build_config(self.argument_name)
        for k, v in parsed.items():
            if isinstance(v, core.Config):
                for prop_name, prop_value in v.get_all(k).items():
                    conf.set(f'{self.argument_name}.{k}.{prop_name}', prop_value)
            else:
                conf.set(f'{self.argument_name}.{k}', v)
        return conf

    @staticmethod
    def _eval_is_enabled(argument: Argument,
                         parsed: dict[str, typing.Any],
                         logger: logging.Logger) -> bool:
        try:
            is_enabled = argument.is_enabled(core.ConfigFactory.dict_config(parsed))
        except core.NotSet as e:
            message = f'Unexpected error: {e}'
            logger.error(message)
            raise RuntimeError(f'Unexpected error: {e}') from e
        except core.NoSuchSetting as e:
            message = (
                f'Error while evaluating on/off status for argument {argument.argument_name}: '
                f'Attempted to retrieve non-existent value {e.attribute}.'
            )
            logger.error(message)
            raise ValueError(message) from e
        except core.IllegalNamespace as e:
            message = (
                f'Error while evaluating on/off status for argument {argument.argument_name}: '
                f'Invalid namespace access while accessing {e.attribute}.'
            )
            logger.error(message)
            raise ValueError(message) from e
        return is_enabled

    def _impose_constraints(self, conf: core.Config):
        for constraint in self._constraints:
            if not constraint.impose(conf):
                raise ArgumentParsingError(f'Constraint {constraint} failed')


def _fmt_list(x: typing.Iterable[str]):
    x = list(x)
    if len(x) == 1:
        return [x]
    return ', '.join(x[:-1]) + f', and {x[-1]}'
