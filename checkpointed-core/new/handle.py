from __future__ import annotations

import functools


@functools.total_ordering
class PipelineStepHandle:

    def __init__(self, uid: int, name: str | None):
        self._uid = uid
        self._name = name

    def __repr__(self):
        return f'{self.__class__.__name__}(uid={self._uid}, name={self._name})'

    def __str__(self):
        if self._name is not None:
            return f'Step<{self._uid}:{self._name}>'
        return f'Step<{self._uid}>'

    @property
    def name(self) -> str | None:
        return self._name

    def get_raw_identifier(self) -> int:
        return self._uid

    def __eq__(self, other):
        if isinstance(other, PipelineStepHandle):
            return self._uid == other._uid
        return False

    def __lt__(self, other):
        if isinstance(other, PipelineStepHandle):
            return self._uid < other._uid
        return NotImplemented

    def __hash__(self):
        return hash(self._uid)
