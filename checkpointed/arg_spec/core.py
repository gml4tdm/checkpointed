from __future__ import annotations

import copy
import typing


class NoSuchSetting(LookupError):
    def __init__(self, attribute, action):
        message = (
            f"Cannot perform action {action!r} on setting "
            f"{attribute!r} since it does not exist"
        )
        self.attribute = attribute
        self.action = action
        super().__init__(message)


class NotSet(Exception):
    def __init__(self, attribute):
        self.attribute = attribute
        message = f"Attribute {attribute!r} has not been initialized"
        super().__init__(message)


class IllegalNamespace(Exception):
    def __init__(self, attribute):
        self.attribute = attribute
        message = f"The namespace containing {attribute!r} is currently not accessible"
        super().__init__(message)


class ConfigFactory:

    def __init__(self):
        self._namespace = {}

    @classmethod
    def dict_config(cls, d, *, namespace=None):
        self = cls()
        no_namespace = namespace is None
        if no_namespace:
            namespace = '$dict'
        self.register_namespace(namespace)
        for key in d:
            self.register(f'{namespace}.{key}')
        if no_namespace:
            config = self._build_dict_config(namespace)
            for key, value in d.items():
                config.set(key, value)
        else:
            config = self.build_config(namespace)
            for key, value in d.items():
                config.set(f'{namespace}.{key}', value)
        return config

    @staticmethod
    def _normalize_name(x: str) -> str:
        return x.lower().replace("-", "_")

    def register_namespace(self, name: str):
        if not name:
            raise ValueError("Name must be non-empty")
        parts = self._normalize_name(name).split(".")
        current = self._namespace
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
            if current is None:
                raise ValueError(
                    f"{name} is a property, and cannot be (part of) a namespace"
                )

    def register(self, name: str):
        if not name:
            raise ValueError("Name must be non-empty")
        parts = self._normalize_name(name).split(".")
        if len(parts) < 2:
            raise ValueError(
                f"A property must be contained in the non-global namespace ({name})"
            )
        prop_name = parts[-1]
        target = self._resolve_property_namespace(parts[:-1], prop_name, name)
        target[prop_name] = None

    def mount_sub_config(self, path: str, other: ConfigFactory):
        parts = self._normalize_name(path).split(".")
        target = self._resolve_property_namespace(parts[:-1], parts[-1], path)
        target[parts[-1]] = copy.deepcopy(other._namespace)

    def build_config(self, *namespaces) -> Config:
        return Config(*self._prepare_config(*namespaces))

    def _resolve_property_namespace(self,
                                    location: list[str],
                                    prop_name: str,
                                    full_path: str):
        target = self._resolve_parent_namespace(location, full_path)
        if prop_name in target and target[prop_name] is not None:
            raise ValueError(
                f"Cannot register property {full_path}; already defined as a namespace"
            )
        if prop_name in target:
            raise ValueError(
                f"Cannot register property {full_path}; property already exists"
            )
        return target

    def _resolve_parent_namespace(self,
                                  location: list[str],
                                  full_path: str) -> dict[str, dict | None]:
        current = self._namespace
        for part in location:
            try:
                current = current[part]
            except KeyError:
                raise ValueError(f'Undefined namespace: {part} (in {full_path})')
            if current is None:
                raise ValueError(f'Path component is a property: {part} (in {full_path})')
        return current

    def _build_dict_config(self, namespace) -> _ConfigDictProxy:
        return _ConfigDictProxy(*self._prepare_config(namespace), prefix=namespace)

    def _prepare_config(self, *namespaces):
        legal = [self._normalize_name(n) for n in namespaces]
        for n in legal:
            if "." in n:
                raise ValueError(
                    f"Can only register top-level namespaces as legal, not {n}"
                )
        return legal, self._namespace, self._new_namespace_tree(self._namespace)

    def _new_namespace_tree(self, obj):
        if obj is None:
            return Config.NOT_SET
        return {key: self._new_namespace_tree(value) for key, value in obj.items()}


T = typing.TypeVar('T')


class Config:
    NOT_SET = object()

    def __init__(self, legal_namespaces, namespaces, data):
        self._legal = legal_namespaces
        self._namespaces = namespaces
        self._data = data

    def _normalize_name(self, x):
        return x.lower().replace("-", "_").split(".")

    def _resolve(self, name, action, path):
        if path[0] not in self._legal:
            raise IllegalNamespace(name)
        current_n = self._namespaces
        current_d = self._data
        for part in path:
            if current_n is None:
                raise NoSuchSetting(name, action)
            if part not in current_n:
                raise NoSuchSetting(name, action)
            current_n = current_n[part]
            current_d = current_d[part]
        return current_d

    def get_all(self, name: str):
        return {
            key: (value if value is not self.NOT_SET else None)
            for key, value in self._resolve(
                name, "get_all", self._normalize_name(name)
            ).items()
        }

    def get(self, name: str) -> typing.Any:
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, "get", path)
        if prop not in namespace:
            raise NoSuchSetting(name, "get")
        value = namespace[prop]
        if value is self.NOT_SET:
            raise NotSet(name)
        return value

    def get_casted(self, name: str, typ: type[T]) -> T:
        value = self.get(name)
        assert isinstance(value, typ), f'Type mismatch: {value.__class__.__name__} != {typ.__name__}'
        return value

    def set(self, name: str, value: typing.Any):
        *path, prop = self._normalize_name(name)
        namespace = self._resolve(name, "set", path)
        if prop not in namespace:
            raise NoSuchSetting(name, "set")
        namespace[prop] = value

    def clone(self, from_: str, to: str):
        self.set(to, self.get(from_))

    def transfer(self, target: Config, *properties):
        for prop in properties:
            target.set(prop, self.get(prop))

    def update(self, prefix: str | None = None, /, **items):
        for key, value in items.items():
            if prefix is not None:
                key = f"{prefix}.{key}"
            self.set(key, value)


class _ConfigDictProxy(Config):

    def __init__(self, legal_namespaces, namespaces, data, *, prefix):
        super().__init__(legal_namespaces, namespaces, data)
        self._prefix = prefix

    def _normalize_name(self, x):
        return [self._prefix, x.replace('-', '_')]
