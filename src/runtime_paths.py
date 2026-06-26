"""Helpers for locating resources and writable runtime directories."""

from pathlib import Path
import os
import sys


def is_frozen():
    """Return True when the app runs from a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def bundle_root():
    """Return the directory that contains bundled read-only resources."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return Path(__file__).resolve().parent


def app_root():
    """Return the writable application directory for artifacts."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(*parts):
    """Resolve a bundled resource regardless of source or packaged mode."""
    return bundle_root().joinpath(*parts)


def artifacts_root():
    """Resolve where weights and logs should be created at runtime."""
    override = os.environ.get("GOMOKU_ARTIFACTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return app_root() / "artifacts"


def bundled_default_weights_path():
    """Return the bundled default weights file when one exists."""
    defaults_dir = resource_path("defaults")
    if not defaults_dir.exists():
        return None

    named_default = defaults_dir / "ppo_rl_latest.pth"
    if named_default.exists():
        return named_default

    candidates = sorted(defaults_dir.glob("*.pth"))
    if candidates:
        return candidates[0]
    return None
