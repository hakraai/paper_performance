from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


T = TypeVar("T")
LOGGER_NAME = "paper_performance"


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    if not name:
        return logging.getLogger(LOGGER_NAME)
    suffix = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")


def progress(iterable: Iterable[T], *, desc: str, total: int | None = None, leave: bool = False) -> Iterator[T]:
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=leave)


@contextmanager
def logging_progress() -> Iterator[None]:
    root_logger = logging.getLogger(LOGGER_NAME)
    with logging_redirect_tqdm(loggers=[root_logger]):
        yield
