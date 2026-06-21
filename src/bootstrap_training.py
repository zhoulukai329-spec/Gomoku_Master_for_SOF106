"""Self-play bootstrap training before launching human-vs-AI play."""

from argparse import Namespace
from shutil import copy2

from storage import DEFAULT_RUN_NAME, latest_weights_path, resolve_weights_path
from train import train


DEFAULT_BOOTSTRAP_EPISODES = 500


def build_bootstrap_args(size, episodes, run_name, resume_path):
    """Create the training args used for automatic pre-game self-play."""
    episodes = max(1, int(episodes))
    return Namespace(
        size=size,
        episodes=episodes,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=1.0,
        update_every=4,
        save_every=max(episodes, 1),
        eval_every=max(episodes, 1),
        eval_games=2,
        eval_temperature=0.05,
        exploration_moves=12,
        exploration_temperature=1.0,
        endgame_temperature=0.2,
        rolling_window=50,
        log_every=max(1, episodes // 10),
        run_name=run_name,
        weights_path=str(resume_path),
        resume=resume_path.exists(),
    )


def ensure_weights_for_play(
    weights_path="",
    run_name=DEFAULT_RUN_NAME,
    size=15,
    episodes=DEFAULT_BOOTSTRAP_EPISODES,
    force_train=False,
):
    """Train by self-play when no playable weights are available."""
    target_path = resolve_weights_path(weights_path, run_name)
    if target_path.exists() and not force_train:
        return target_path, False

    latest_path = latest_weights_path(run_name)
    resume_path = target_path if target_path.exists() else latest_path
    args = build_bootstrap_args(size, episodes, run_name, resume_path)

    if force_train and target_path.exists():
        print(f"[Bootstrap] Continuing self-play training from {target_path}")
    elif latest_path.exists():
        print(f"[Bootstrap] No target weights found. Continuing from {latest_path}")
    else:
        print("[Bootstrap] No weights found. Running self-play training before play.")

    train(args)

    trained_path = latest_weights_path(run_name)
    if target_path != trained_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(trained_path, target_path)
    return target_path, True
