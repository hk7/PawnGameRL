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

        # Randomized Openings for Training Diversity ---
        if options and options.get("scramble_board", False):
            import random
            # Play a random number of opening moves (between 2 and 6) to mix up positions
            num_scramble_moves = random.randint(2, 6)
            for _ in range(num_scramble_moves):
                legal_moves = list(self.board.legal_moves)
                if not legal_moves or self.board.is_game_over():
                    break
                # Filter out promotions during scramble to keep it clean
                non_promo_moves = [m for m in legal_moves if not (self.board.piece_at(m.from_square).piece_type == chess.PAWN and chess.square_rank(m.to_square) in [0, 7])]
                if non_promo_moves:
                    self.board.push(random.choice(non_promo_moves))

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
        """Converts a flattened action index back into a chess.Move object safely."""
        from_square = action_idx // 64
        to_square = action_idx % 64

        move = chess.Move(from_square, to_square)

        # Only apply promotion if the piece moving is actually a PAWN
        piece = self.board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if chess.square_rank(to_square) in [0, 7]:
                move.promotion = chess.QUEEN

        return move


    def old_step(self, action_idx):
        move = self.decode_action(action_idx)

        if move not in self.board.legal_moves:
            logger.error(f"Illegal action index attempted: {action_idx} (Move: {move})")
            raise ValueError(f"Illegal move attempted: {move}")

        # Get the color of the player making the move RIGHT NOW
        current_turn_color = self.board.turn

        moving_piece = self.board.piece_at(move.from_square)
        is_pawn_promotion = (
            moving_piece and 
            moving_piece.piece_type == chess.PAWN and 
            chess.square_rank(move.to_square) in [0, 7]
        )

        # Execute move (flips the turn automatically)
        self.board.push(move)

        terminated = False
        reward = 0.0

        if is_pawn_promotion:
            terminated = True
            reward = 10.0 if current_turn_color == chess.WHITE else -10.0
            logger.info(f"Pawn promotion achieved by {'White' if current_turn_color == chess.WHITE else 'Black'} on {chess.square_name(move.to_square)}!")
        else:
            # Check if the NEXT player has any moves left
            next_player_has_moves = bool(list(self.board.legal_moves))

            # Check for immobilization / no legal moves ---
            if not next_player_has_moves:
                terminated = True
                # The player whose turn it is now has no moves, meaning they LOSE.
                # If it's White's turn and they are blocked -> Black wins (-10.0)
                # If it's Black's turn and they are blocked -> White wins (+10.0)
                reward = -10.0 if self.board.turn == chess.WHITE else 10.0
                logger.info(f"Game over by block! {'White' if self.board.turn == chess.WHITE else 'Black'} is out of legal moves.")

            elif self.board.is_game_over():
                # Fallback for standard chess termination conditions if any apply
                terminated = True
                outcome = self.board.outcome()
                if outcome and outcome.winner == chess.WHITE:
                    reward = 10.0
                elif outcome and outcome.winner == chess.BLACK:
                    reward = -10.0

            # INTERMEDIATE MATERIAL REWARDS
            if not terminated:
                # Count current pieces on the board
                white_pawns = len(self.board.pieces(chess.PAWN, chess.WHITE))
                black_pawns = len(self.board.pieces(chess.PAWN, chess.BLACK))
                
                # We want White to maximize this score, and Black to minimize it
                # Give a tiny weight (0.2) so it doesn't override the main goal of promotion
                material_balance = (white_pawns - black_pawns) * 0.2
                
                # If it's White's training turn, a positive balance is good
                reward += material_balance

        return self._get_obs(), reward, terminated, False, {}


    def step(self, action_idx):
        move = self.decode_action(action_idx)

        if move not in self.board.legal_moves:
            logger.error(f"Illegal action index attempted: {action_idx} (Move: {move})")
            raise ValueError(f"Illegal move attempted: {move}")

        # Get the color of the player making the move right now
        current_turn_color = self.board.turn

        moving_piece = self.board.piece_at(move.from_square)
        is_pawn_promotion = (
            moving_piece and 
            moving_piece.piece_type == chess.PAWN and 
            chess.square_rank(move.to_square) in [0, 7]
        )

        # Execute move (flips the turn automatically)
        self.board.push(move)

        terminated = False
        base_reward = 0.0

        # 1. Check Win / Loss / Block conditions
        if is_pawn_promotion:
            terminated = True
            base_reward = 10.0 if current_turn_color == chess.WHITE else -10.0
            logger.info(f"Pawn promotion achieved by {'White' if current_turn_color == chess.WHITE else 'Black'}!")
        else:
            next_player_has_moves = bool(list(self.board.legal_moves))
            if not next_player_has_moves:
                terminated = True
                # The trapped player loses. If White is trapped -> Black wins (-10). If Black trapped -> White wins (+10)
                base_reward = -10.0 if self.board.turn == chess.WHITE else 10.0
                logger.info(f"Game over by block! {'White' if self.board.turn == chess.WHITE else 'Black'} is trapped.")
            elif self.board.is_game_over():
                terminated = True
                outcome = self.board.outcome()
                if outcome and outcome.winner == chess.WHITE:
                    base_reward = 10.0
                elif outcome and outcome.winner == chess.BLACK:
                    base_reward = -10.0

        # 2. Add Material Balance Shaping (Crucial for tactical awareness)
        # We calculate the count of pieces left on the board
        white_count = len(self.board.pieces(chess.PAWN, chess.WHITE))
        black_count = len(self.board.pieces(chess.PAWN, chess.BLACK))

        # Every pawn is worth 1.0 point of intermediate shaping reward
        material_shaping = (white_count - black_count) * 1.0

        # Total reward combines the game status with the material standing
        reward = base_reward + material_shaping

        return self._get_obs(), reward, terminated, False, {}



    def action_masks(self):
        """Returns a boolean array of size 4096 where True means the move is legal."""
        mask = np.zeros(4096, dtype=bool)
        for move in self.board.legal_moves:
            # Replicate the implicit promotion logic to stay consistent with decode_action
            moving_piece = self.board.piece_at(move.from_square)
            if moving_piece and moving_piece.piece_type == chess.PAWN:
                if chess.square_rank(move.to_square) in [0, 7]:
                    move.promotion = chess.QUEEN
                    
            action_idx = move.from_square * 64 + move.to_square
            mask[action_idx] = True
        return mask



    def render(self):
        """Renders the board with high-visibility symbols and clear file/rank borders."""
        # Mapping chess pieces to clear Unicode symbols
        unicode_pieces = {
            'P': '♙',  # White Pawn
            'p': '♟',  # Black Pawn
            '.': '·'   # Empty square
        }

        board_str = str(self.board)
        lines = board_str.split('\n')

        print("\n  a b c d e f g h")
        print("  ---------------")
        for i, line in enumerate(lines):
            rank = 8 - i
            # Replace characters with high-visibility symbols
            formatted_line = " ".join([unicode_pieces.get(char, char) for char in line.split()])
            print(f"{rank}|{formatted_line}|{rank}")
        print("  ---------------")
        print("  a b c d e f g h\n")
