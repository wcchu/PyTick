"""Command-line interface for the PyTick training and play flows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agents import HumanAgent, MinimaxAgent, QLearningAgent, RandomAgent
from .board import Board
from .training import evaluate_agents, play_game, train_q_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the PyTick tic-tac-toe agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the Q-learning agent.")
    train_parser.add_argument("--episodes", type=int, default=2000)
    train_parser.add_argument("--output", type=str, default="results/training_metrics.csv")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--opponent", choices=["random", "minimax"], default="random")

    play_parser = subparsers.add_parser("play", help="Play a game against an opponent.")
    play_parser.add_argument("--x", choices=["human", "random", "minimax", "q"], default="human")
    play_parser.add_argument("--o", choices=["human", "random", "minimax", "q"], default="random")
    play_parser.add_argument("--render", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the trained agent.")
    evaluate_parser.add_argument("--episodes", type=int, default=200)
    evaluate_parser.add_argument("--seed", type=int, default=7)

    return parser


def create_agent(kind: str, mark: int):
    if kind == "human":
        return HumanAgent(mark)
    if kind == "random":
        return RandomAgent(mark)
    if kind == "minimax":
        return MinimaxAgent(mark)
    if kind == "q":
        return QLearningAgent(mark)
    raise ValueError(f"Unsupported agent type: {kind}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        metrics = train_q_agent(
            episodes=args.episodes,
            output_path=args.output,
            seed=args.seed,
            opponent=args.opponent,
        )
        print(f"Saved {len(metrics)} training records to {Path(args.output)}")
        return 0

    if args.command == "play":
        agent_x = create_agent(args.x, Board.X)
        agent_o = create_agent(args.o, Board.O)
        result = play_game(agent_x, agent_o, render=args.render)
        winner = result["winner"]
        if winner == Board.X:
            print(f"X wins in {result['moves']} moves.")
        elif winner == Board.O:
            print(f"O wins in {result['moves']} moves.")
        else:
            print(f"Draw after {result['moves']} moves.")
        return 0

    if args.command == "evaluate":
        agent_x = QLearningAgent(mark=Board.X)
        agent_o = RandomAgent(mark=Board.O)
        summary = evaluate_agents(agent_x, agent_o, episodes=args.episodes, seed=args.seed)
        print(summary)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
