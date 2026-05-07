"""utils/helpers.py — logging, config, JSON, timing."""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any, Dict
import numpy as np, yaml


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    return log


def load_config(path: str | Path) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):              return int(o)
    if isinstance(o, (np.floating,)):             return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):                 return o.tolist()
    if isinstance(o, float) and np.isnan(o):      return None
    raise TypeError(type(o))


class Timer:
    def __enter__(self):  self._t = time.perf_counter(); return self
    def __exit__(self,*_): self.elapsed = time.perf_counter() - self._t
