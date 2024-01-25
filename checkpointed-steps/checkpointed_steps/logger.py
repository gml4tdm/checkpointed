import datetime
import logging
import os
import sys


def default_logger(name: str, level, *,
                   stdout=False,
                   stderr=True,
                   stdout_level=logging.INFO,
                   stderr_level=logging.DEBUG,
                   formatter=None,
                   logging_directory: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if formatter is None:
        formatter = logging.Formatter(
            '[{asctime}][{levelname:<8}][{name}]: {message}',
            style='{'
        )
    if stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(stderr_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if logging_directory is not None:
        os.makedirs(logging_directory, exist_ok=True)
        date = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        handler = logging.FileHandler(f'{logging_directory}/{date}.log')
        handler.setLevel(stderr_level)
        handler.setFormatter(formatter)
    return logger
