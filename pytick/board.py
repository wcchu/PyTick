"""Core tic-tac-toe board logic for the PyTick project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Board:
    """Immutable-by-convention board representing a standard 3x3 game."""

    EMPTY = 0
    X = 1
    O = -1

    cells: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0])
    current_player: int = X

    def copy(self) -> "Board":
        return Board(cells=self.cells[:], current_player=self.current_player)

    def available_moves(self) -> List[int]:
        return [idx for idx, value in enumerate(self.cells) if value == self.EMPTY]

    def make_move(self, index: int, player: Optional[int] = None) -> None:
        if index < 0 or index >= len(self.cells):
            raise ValueError(f"Move index must be between 0 and {len(self.cells) - 1}.")
        if self.cells[index] != self.EMPTY:
            raise ValueError(f"Square {index} is already occupied.")

        resolved_player = self.current_player if player is None else player
        if resolved_player not in (self.X, self.O):
            raise ValueError("Player must be 1 (X) or -1 (O).")

        self.cells[index] = resolved_player
        self.current_player = -resolved_player

    def winner(self) -> int:
        lines = [
            self.cells[0:3],
            self.cells[3:6],
            self.cells[6:9],
            self.cells[0:9:3],
            self.cells[1:9:3],
            self.cells[2:9:3],
            self.cells[0:9:4],
            self.cells[2:7:2],
        ]
        for line in lines:
            if line[0] != self.EMPTY and line[0] == line[1] == line[2]:
                return line[0]
        return self.EMPTY

    def terminal(self) -> bool:
        return self.winner() != self.EMPTY or not self.available_moves()

    def is_draw(self) -> bool:
        return self.winner() == self.EMPTY and not self.available_moves()

    def to_tuple(self) -> Tuple[int, ...]:
        return tuple(self.cells)

    def display(self) -> str:
        rows = []
        for row in range(3):
            values = []
            for col in range(3):
                idx = row * 3 + col
                cell = self.cells[idx]
                if cell == self.X:
                    value = "X"
                elif cell == self.O:
                    value = "O"
                else:
                    value = str(idx)
                values.append(value)
            rows.append(" | ".join(values))
        return "\n---\n".join(rows)

    def __str__(self) -> str:
        return self.display()

    def __repr__(self) -> str:
        return f"Board(cells={self.cells}, current_player={self.current_player})"
