"""
Face subsystem configuration loader.

Loads face_config.toml once at import time. Keep this file tiny — it's
imported by agent.py at startup, so no heavy dependencies.
"""

import os

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "face_config.toml")
    with open(path, "rb") as f:
        return tomllib.load(f)


_CONFIG = _load_config()


def get_tracker_config() -> dict:
    return _CONFIG.get("tracker", {})
