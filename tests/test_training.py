import unittest

from pytick.board import Board
from pytick.training import train_q_agent


class TrainingTests(unittest.TestCase):
    def test_train_q_agent_outputs_metrics(self):
        metrics = train_q_agent(episodes=20, output_path="results/test_training_metrics.csv", seed=11)
        self.assertEqual(len(metrics), 20)
        self.assertEqual(metrics[0]["episode"], 1)
        self.assertTrue(0.0 <= metrics[-1]["win_rate"] <= 1.0)

    def test_board_can_terminate(self):
        board = Board(cells=[1, 1, 1, -1, -1, 0, 0, 0, 0])
        self.assertTrue(board.terminal())
        self.assertEqual(board.winner(), Board.X)


if __name__ == "__main__":
    unittest.main()
