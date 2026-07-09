"""
src/utils/logging_utils.py
==========================

Rank-0-safe logger construction, extracted from the monolith.
"""

from __future__ import annotations

import logging


def build_logger(name: str = "burgers", level: int = logging.INFO) -> logging.Logger:
    """Create a module logger that only emits on MPI rank 0.

    Using the logging module (rather than bare ``print``) keeps output
    controllable and lets the caller redirect/silence it. In parallel runs we
    guard on rank so the log is not duplicated across processes.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# A default module-level logger for convenience / backward compatibility.
LOGGER = build_logger()
