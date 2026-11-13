"""Summarize final training metrics exported by the PyTick Q-learning workflow."""

from __future__ import annotations

import csv
from pathlib import Path


def load_metrics(path: str | Path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    input_path = Path("results/training_metrics.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Training metrics file not found: {input_path}")

    metrics = load_metrics(input_path)
    if not metrics:
        raise ValueError(f"No metrics available in {input_path}")

    last = metrics[-1]
    summary = {
        "final_win_rate": float(last["win_rate"]),
        "final_draw_rate": float(last["draw_rate"]),
        "final_loss_rate": float(last["loss_rate"]),
        "final_epsilon": float(last["epsilon"]),
    }

    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
