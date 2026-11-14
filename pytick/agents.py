"""Agent implementations for tic-tac-toe players."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .board import Board


class BaseAgent:
    def __init__(self, mark: int, name: str):
        self.mark = mark
        self.name = name

    def choose_action(self, board: Board) -> Optional[int]:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


class HumanAgent(BaseAgent):
    def __init__(self, mark: int):
        super().__init__(mark, "Human")

    def choose_action(self, board: Board) -> Optional[int]:
        available = board.available_moves()
        if not available:
            return None

        while True:
            try:
                raw = input(f"Player {self.mark} pick a square (0-8): ").strip()
                move = int(raw)
                if move in available:
                    return move
                print(f"Invalid move. Available moves: {available}")
            except ValueError:
                print("Please enter an integer from 0 to 8.")


class RandomAgent(BaseAgent):
    def __init__(self, mark: int):
        super().__init__(mark, "Random")

    def choose_action(self, board: Board) -> Optional[int]:
        available = board.available_moves()
        if not available:
            return None
        return random.choice(available)


class MinimaxAgent(BaseAgent):
    def __init__(self, mark: int):
        super().__init__(mark, "Minimax")

    def choose_action(self, board: Board) -> Optional[int]:
        available = board.available_moves()
        if not available:
            return None

        best_score = -float("inf")
        best_move = available[0]
        for move in available:
            next_board = board.copy()
            next_board.make_move(move, self.mark)
            score = self._minimax(next_board, depth=1, maximizing_player=self.mark == Board.X)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _minimax(self, board: Board, depth: int, maximizing_player: bool) -> int:
        winner = board.winner()
        if winner == self.mark:
            return 10 - depth
        if winner == -self.mark:
            return depth - 10
        if board.is_draw():
            return 0

        available = board.available_moves()
        if maximizing_player:
            best_score = -float("inf")
            for move in available:
                next_board = board.copy()
                next_board.make_move(move, self.mark)
                score = self._minimax(next_board, depth + 1, False)
                best_score = max(best_score, score)
            return best_score

        best_score = float("inf")
        opponent_mark = -self.mark
        for move in available:
            next_board = board.copy()
            next_board.make_move(move, opponent_mark)
            score = self._minimax(next_board, depth + 1, True)
            best_score = min(best_score, score)
        return best_score


class QLearningAgent(BaseAgent):
    def __init__(self, mark: int, epsilon: float = 0.1, alpha: float = 0.2, gamma: float = 0.9):
        super().__init__(mark, "QLearning")
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: Dict[Tuple[int, ...], Dict[int, float]] = {}

    def _state_key(self, board: Board) -> Tuple[int, ...]:
        return board.to_tuple()

    def choose_action(self, board: Board) -> Optional[int]:
        available = board.available_moves()
        if not available:
            return None

        if random.random() < self.epsilon:
            return random.choice(available)

        key = self._state_key(board)
        q_values = self.q_table.get(key, {})
        if not q_values:
            return random.choice(available)

        best_move = max(available, key=lambda move: q_values.get(move, 0.0))
        return best_move

    def learn(self, previous_state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...], done: bool) -> None:
        previous_q = self.q_table.setdefault(previous_state, {})
        next_q = self.q_table.setdefault(next_state, {})

        current_value = previous_q.get(action, 0.0)
        future_value = max(next_q.values(), default=0.0) if not done else 0.0
        target = reward + self.gamma * future_value
        previous_q[action] = current_value + self.alpha * (target - current_value)

    def update_epsilon(self, value: float) -> None:
        self.epsilon = max(0.0, min(1.0, value))
