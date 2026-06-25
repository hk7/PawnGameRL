import logging
import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("Environment")

class PawnGameEnv(gym.Env):
    """
    A custom Gymnasium environment for the Chess Pawn Game.
    White pawns start on rank 2, Black pawns on rank 7.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8, 8), dtype=np.float32)
        self.action_space = spaces.Discrete(4096)
        self.board = chess.Board(fen=None)
        logger.debug("Environment initialized successfully.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board.clear()
        
        # Set up Pawns on the 2nd (White) and 7th (Black) ranks
        for file in range(8):
            self.board.set_piece_at(chess.square(file, 1), chess.Piece(chess.PAWN, chess.WHITE))
            self.board.set_piece_at(chess.square(file, 6), chess.Piece(chess.PAWN, chess.BLACK))
        
        logger.info("Game environment reset. Board is ready.")
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros((8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                row, col = divmod(square, 8)
                obs[row, col] = 1.0 if piece.color == chess.WHITE else -1.0
        return obs

    def get_legal_actions(self):
        """Returns a list of legal moves converted to flattened action indices."""
        actions = []
        for move in self.board.legal_moves:
            action_idx = move.from_square * 64 + move.to_square
            actions.append(action_idx)
        return actions

    def decode_action(self, action_idx):
        """Converts a flattened action index back into a chess.Move object."""
        from_square = action_idx // 64
        to_square = action_idx % 64
        
        move = chess.Move(from_square, to_square)
        if chess.square_rank(to_square) in [0, 7]:
            move.promotion = chess.QUEEN
        return move

    def step(self, action_idx):
        move = self.decode_action(action_idx)
        
        if move not in self.board.legal_moves:
            logger.error(f"Illegal action index attempted: {action_idx} (Move: {move})")
            raise ValueError(f"Illegal move attempted: {move}")

        # Execute move
        self.board.push(move)
        
        # Check game-over conditions
        terminated = self.board.is_game_over()
        reward = 0.0

        if terminated:
            outcome = self.board.outcome()
            if outcome.winner == chess.WHITE:
                reward = 10.0
            elif outcome.winner == chess.BLACK:
                reward = -10.0
        
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        # Using print for the visible board grid, logging handles metadata
        print("\n" + str(self.board) + "\n")
        