"""PyTick: a Python tic-tac-toe reinforcement learning project."""

from .agents import HumanAgent, MinimaxAgent, QLearningAgent, RandomAgent
from .board import Board
from .training import evaluate_agents, train_q_agent

__all__ = [
    "Board",
    "HumanAgent",
    "MinimaxAgent",
    "QLearningAgent",
    "RandomAgent",
    "evaluate_agents",
    "train_q_agent",
]

__version__ = "0.1.0"
