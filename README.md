# PyTick

PyTick is a small Python port of the GoTick tic-tac-toe reinforcement-learning project. The game engine, training logic, CLI, and analysis scripts are all implemented in Python.

## Features

- Board logic for standard 3x3 tic-tac-toe
- Human, random, minimax, and Q-learning agents
- Training loop that exports cumulative metrics to CSV
- CLI to train, evaluate, and play games
- Python analysis scripts to summarize and plot the exported training data

## Quick start

```bash
python3 -m pytick train --episodes 2000 --output results/training_metrics.csv --seed 42
python3 -m pytick play --x q --o random --render
python3 -m pytick evaluate --episodes 100 --seed 7
python3 analysis/plot_training.py
python3 analysis/summarize_training.py
```

## Project layout

- `pytick/` – Python package
- `analysis/` – Python-based analysis scripts
- `results/` – training output files (CSV, plots)
- `tests/` – lightweight verification tests

## Analysis

The analysis scripts read the metrics saved by the training workflow and generate a summary or plot.

```bash
python3 analysis/plot_training.py
python3 analysis/summarize_training.py
```

This reads the metrics from `results/training_metrics.csv` and writes a plot to `results/training_summary.png`.
