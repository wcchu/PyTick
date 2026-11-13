"""Plot training metrics exported by the PyTick Q-learning workflow."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: str | Path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    input_path = Path("results/training_metrics.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Training metrics file not found: {input_path}")

    metrics = load_metrics(input_path)
    if not metrics:
        raise ValueError(f"No training metrics available in {input_path}")

    episodes = [int(row["episode"]) for row in metrics]
    wins = [float(row["win_rate"]) for row in metrics]
    draws = [float(row["draw_rate"]) for row in metrics]
    losses = [float(row["loss_rate"]) for row in metrics]
    rolling = []
    window = min(50, len(metrics))
    for idx in range(len(metrics)):
        start = max(0, idx - window + 1)
        rolling_window = wins[start : idx + 1]
        rolling.append(sum(rolling_window) / len(rolling_window))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, wins, label="Win rate", color="steelblue", linewidth=2)
    ax.plot(episodes, draws, label="Draw rate", color="darkgreen", linewidth=2)
    ax.plot(episodes, losses, label="Loss rate", color="firebrick", linewidth=2)
    ax.plot(episodes, rolling, label="Rolling win rate", color="purple", linewidth=2, linestyle="--")

    ax.set_title("PyTick training performance")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    output_path = Path("results/training_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved plot to {output_path}")
    print("Summary:")
    for key in ("win_rate", "draw_rate", "loss_rate", "epsilon"):
        values = [float(row[key]) for row in metrics]
        print(f"  {key}: min={min(values):.4f}, max={max(values):.4f}, mean={sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
