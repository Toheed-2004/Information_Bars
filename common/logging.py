"""
common/logging.py
-----------------
Lightweight logging — no external dependencies.
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a console logger for *name*."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once at startup."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
