from __future__ import annotations


class PipelineStepHandle:

    def __init__(self, uid: int):
        self._uid = uid

    def get_raw_identifier(self) -> int:
        return self._uid

    def __eq__(self, other):
        if isinstance(other, PipelineStepHandle):
            return self._uid == other._uid
        return False

    def __hash__(self):
        return hash(self._uid)
