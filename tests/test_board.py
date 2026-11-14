import unittest

from pytick.board import Board


class BoardTests(unittest.TestCase):
    def test_winner_on_row(self):
        board = Board(cells=[1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(board.winner(), Board.X)

    def test_winner_on_diagonal(self):
        board = Board(cells=[-1, 0, 0, 0, -1, 0, 0, 0, -1])
        self.assertEqual(board.winner(), Board.O)

    def test_available_moves(self):
        board = Board(cells=[1, 0, 1, 0, -1, 0, 0, 0, 0])
        self.assertEqual(board.available_moves(), [1, 3, 5, 6, 7, 8])

    def test_make_move_updates_turn(self):
        board = Board()
        board.make_move(0, Board.X)
        self.assertEqual(board.cells[0], Board.X)
        self.assertEqual(board.current_player, Board.O)


if __name__ == "__main__":
    unittest.main()
