from __future__ import annotations


class ArgumentParsingError(Exception):

    def __init__(self, message: str, *, is_user_error=True):
        super().__init__(message)
        self.message = message
        self.is_user_error = is_user_error
