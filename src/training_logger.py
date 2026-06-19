import csv
import json
from collections import Counter, deque

from storage import training_log_path, training_summary_path


class RollingStats:
    """Track recent results so the trainer can report moving win rates."""

    def __init__(self, window_size):
        # Keep only the newest N results inside the rolling window.
        self._window = deque(maxlen=window_size)

    def add_result(self, winner):
        """Append one finished game result to the rolling window."""
        self._window.append(winner)

    def rates(self):
        """Return black win, white win, and draw rates for the current window."""
        total = len(self._window)
        if total == 0:
            return {
                "black_win_rate": 0.0,
                "white_win_rate": 0.0,
                "draw_rate": 0.0,
            }

        counts = Counter(self._window)
        return {
            "black_win_rate": counts.get(1, 0) / total,
            "white_win_rate": counts.get(2, 0) / total,
            "draw_rate": counts.get(3, 0) / total,
        }


class TrainingLogger:
    """Write structured training metrics to CSV and summary JSON files."""

    FIELDNAMES = [
        "episode",
        "steps",
        "winner",
        "episode_reward",
        "policy_loss",
        "value_loss",
        "entropy",
        "black_win_rate",
        "white_win_rate",
        "draw_rate",
        "eval_win_rate",
        "best_eval_win_rate",
        "checkpoint_path",
    ]

    def __init__(self, run_name, rolling_window=100):
        # Resolve the output files once so later writes stay simple.
        self.run_name = run_name
        self.log_path = training_log_path(run_name)
        self.summary_path = training_summary_path(run_name)
        self.rolling_stats = RollingStats(rolling_window)

        # Start every run with a fresh CSV header.
        with self.log_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log_episode(self, metrics):
        """Append one episode row and return the updated rolling win rates."""
        self.rolling_stats.add_result(metrics["winner"])
        rates = self.rolling_stats.rates()

        # Format numeric values as strings so the CSV stays compact and uniform.
        row = {
            "episode": metrics["episode"],
            "steps": metrics["steps"],
            "winner": metrics["winner"],
            "episode_reward": f"{metrics['episode_reward']:.4f}",
            "policy_loss": f"{metrics['policy_loss']:.6f}",
            "value_loss": f"{metrics['value_loss']:.6f}",
            "entropy": f"{metrics['entropy']:.6f}",
            "black_win_rate": f"{rates['black_win_rate']:.4f}",
            "white_win_rate": f"{rates['white_win_rate']:.4f}",
            "draw_rate": f"{rates['draw_rate']:.4f}",
            "eval_win_rate": f"{metrics['eval_win_rate']:.4f}",
            "best_eval_win_rate": f"{metrics['best_eval_win_rate']:.4f}",
            "checkpoint_path": metrics["checkpoint_path"],
        }

        # Append the current episode after all fields are prepared.
        with self.log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

        return rates

    def write_summary(self, summary):
        """Write the final summary once training has ended."""
        with self.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)
