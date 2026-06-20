import csv
import json
from collections import Counter, deque

from storage import training_log_path, training_summary_path


class RollingStats:
    """Track recent results so the trainer can report moving win rates."""

    def __init__(self, window_size):
        # A deque with maxlen automatically drops the oldest entry once it is full, so this always holds exactly the most recent N results (or fewer, near the start of training) without any manual trimming.
        self._window = deque(maxlen=window_size)

    def add_result(self, winner):
        """Append one finished game result to the rolling window."""
        self._window.append(winner)

    def rates(self):
        """Return black win, white win, and draw rates for the current window."""
        total = len(self._window)
        if total == 0:
            # avoid a divide-by-zero before any games have finished.
            return {
                "black_win_rate": 0.0,
                "white_win_rate": 0.0,
                "draw_rate": 0.0,
            }

        # count how many of 1 (black win), 2 (white win), and 3 (draw) appear in the current window, then convert each count into a fraction of the total games in the window.
        counts = Counter(self._window)
        return {
            "black_win_rate": counts.get(1, 0) / total,
            "white_win_rate": counts.get(2, 0) / total,
            "draw_rate": counts.get(3, 0) / total,
        }


class TrainingLogger:
    """Write structured training metrics to CSV and summary JSON files."""

    # The exact column order used for every row written to the CSV file.
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
        # resolve the output file paths once at construction time, so every later write just reuses self.log_path / self.summary_path.
        self.run_name = run_name
        self.log_path = training_log_path(run_name)
        self.summary_path = training_summary_path(run_name)
        self.rolling_stats = RollingStats(rolling_window)

        # start every run with a brand-new CSV file containing only the header row. This means re-running training with the same run name will overwrite (not append to) any previous log.
        with self.log_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log_episode(self, metrics):
        """Append one episode row and return the updated rolling win rates."""
        # feed this episode's winner into the rolling window first, so the win-rate numbers written to this very row already include the episode that just finished.
        self.rolling_stats.add_result(metrics["winner"])
        rates = self.rolling_stats.rates()

        # format every numeric value as a fixed-precision string so the CSV stays compact, consistent, and easy to read by eye or in a spreadsheet, rather than showing Python's raw float repr.
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

        # open in append mode and write exactly one new row, so the file stays small in memory regardless of how many episodes run.
        with self.log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

        return rates

    def write_summary(self, summary):
        """Write the final summary once training has ended."""
        # A single pretty-printed JSON file with the run's key results, meant to be the one place to look for "how did this run go".
        with self.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=2)