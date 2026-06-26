from datetime import datetime
from pathlib import Path

from runtime_paths import app_root, artifacts_root


# Point saved artifacts to a writable runtime directory.
PROJECT_ROOT = app_root()
ARTIFACTS_DIR = artifacts_root()
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# Use one default run name when the user does not provide a custom label.
DEFAULT_RUN_NAME = "ppo_rl"


def ensure_artifact_dirs():
    """Create the artifact folders when they do not exist yet."""
    for directory in (ARTIFACTS_DIR, WEIGHTS_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def create_run_name(prefix=DEFAULT_RUN_NAME):
    """Build a run name that is easy to sort by time."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def latest_weights_path(run_name=DEFAULT_RUN_NAME):
    """Return the path used for the newest resumable weights."""
    ensure_artifact_dirs()
    return WEIGHTS_DIR / f"{run_name}_latest.pth"


def best_weights_path(run_name=DEFAULT_RUN_NAME):
    """Return the path used for the best evaluation checkpoint."""
    ensure_artifact_dirs()
    return WEIGHTS_DIR / f"{run_name}_best.pth"


def checkpoint_weights_path(run_name, episode):
    """Return the path used for a numbered training checkpoint."""
    ensure_artifact_dirs()
    return WEIGHTS_DIR / f"{run_name}_ep{episode:06d}.pth"


def training_log_path(run_name):
    """Return the CSV path that stores per-episode training metrics."""
    ensure_artifact_dirs()
    return LOGS_DIR / f"{run_name}_training.csv"


def training_summary_path(run_name):
    """Return the JSON path that stores the final run summary."""
    ensure_artifact_dirs()
    return LOGS_DIR / f"{run_name}_summary.json"


def resolve_weights_path(weights_path=None, run_name=DEFAULT_RUN_NAME):
    """Use an explicit weights path when given, otherwise use the default latest path."""
    ensure_artifact_dirs()
    if weights_path:
        return Path(weights_path).expanduser().resolve()
    return latest_weights_path(run_name)
