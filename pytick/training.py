"""Training logic for the reinforcement learning tic-tac-toe agent."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .agents import BaseAgent, QLearningAgent, RandomAgent
from .board import Board


def play_game(agent_x: BaseAgent, agent_o: BaseAgent, render: bool = False) -> Dict[str, object]:
    board = Board()
    current_turn = Board.X

    while not board.terminal():
        if render:
            print(f"\nTurn: {board.current_player}")
            print(board)

        agent = agent_x if current_turn == Board.X else agent_o
        move = agent.choose_action(board)
        if move is None:
            break

        board.make_move(move, current_turn)
        current_turn = -current_turn

    winner = board.winner()
    return {
        "winner": winner,
        "final_board": board,
        "moves": sum(1 for value in board.cells if value != Board.EMPTY),
    }


def evaluate_agents(agent_x: BaseAgent, agent_o: BaseAgent, episodes: int = 100, seed: int | None = None) -> Dict[str, float]:
    random_gen = random.Random(seed)
    wins = 0
    losses = 0
    draws = 0

    for _ in range(episodes):
        board = Board()
        current_turn = Board.X
        while not board.terminal():
            agent = agent_x if current_turn == Board.X else agent_o
            move = agent.choose_action(board)
            if move is None:
                break
            board.make_move(move, current_turn)
            current_turn = -current_turn

        outcome = board.winner()
        if outcome == agent_x.mark:
            wins += 1
        elif outcome == agent_o.mark:
            losses += 1
        else:
            draws += 1

    total = max(episodes, 1)
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
    }


def train_q_agent(
    episodes: int = 2000,
    output_path: str | Path = "results/training_metrics.csv",
    seed: int | None = None,
    opponent: str = "random",
) -> List[Dict[str, float]]:
    random_gen = random.Random(seed)
    q_agent = QLearningAgent(mark=Board.X, epsilon=1.0, alpha=0.2, gamma=0.9)
    opponent_agent = RandomAgent(mark=Board.O)
    stats: List[Dict[str, float]] = []

    if opponent == "minimax":
        from .agents import MinimaxAgent

        opponent_agent = MinimaxAgent(mark=Board.O)

    for episode in range(1, episodes + 1):
        board = Board()
        previous_state = None
        previous_action = None
        episode_reward = 0.0

        while not board.terminal():
            if board.current_player == Board.X:
                previous_state = board.to_tuple()
                action = q_agent.choose_action(board)
                if action is None:
                    break
                previous_action = action
                board.make_move(action, Board.X)
                ended = board.terminal()
                outcome = board.winner()
                if outcome == Board.X:
                    reward = 1.0
                elif outcome == Board.O:
                    reward = -1.0
                elif ended:
                    reward = 0.1
                else:
                    reward = 0.0
                episode_reward += reward
                q_agent.learn(previous_state, action, reward, board.to_tuple(), ended)
            else:
                action = opponent_agent.choose_action(board)
                if action is None:
                    break
                board.make_move(action, Board.O)

        outcome = board.winner()
        if outcome == Board.X:
            result = 1.0
        elif outcome == Board.O:
            result = 0.0
        else:
            result = 0.5

        cumulative_wins = sum(1 for row in stats if row["result"] == 1.0)
        cumulative_losses = sum(1 for row in stats if row["result"] == 0.0)
        cumulative_draws = sum(1 for row in stats if row["result"] == 0.5)

        if outcome == Board.X:
            cumulative_wins += 1
        elif outcome == Board.O:
            cumulative_losses += 1
        else:
            cumulative_draws += 1

        win_rate = cumulative_wins / episode
        loss_rate = cumulative_losses / episode
        draw_rate = cumulative_draws / episode
        q_agent.update_epsilon(max(0.01, 1.0 - ((episode + 1) / (episodes + 1))))

        stats.append(
            {
                "episode": episode,
                "result": result,
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "draw_rate": draw_rate,
                "epsilon": q_agent.epsilon,
                "episode_reward": episode_reward,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["episode", "result", "win_rate", "loss_rate", "draw_rate", "epsilon", "episode_reward"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)

    return stats
