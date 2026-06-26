import abc
import random
import logging
import chess
from environment import PawnGameEnv
from sb3_contrib import MaskablePPO

logger = logging.getLogger("Players")

class Player(abc.ABC):
    """Abstract Base Class for all Pawn Game players."""
    def __init__(self, color: chess.Color, name: str):
        self.color = color
        self.name = name

    @abc.abstractmethod
    def choose_action(self, env: PawnGameEnv) -> int:
        """Must return a legal flattened action index."""
        pass


class RandomPlayer(Player):
    """Selects moves completely at random."""
    def __init__(self, color, name):
        super().__init__(color, name)
        self.is_ai = True

    def choose_action(self, env: PawnGameEnv) -> int:
        legal_actions = env.get_legal_actions()
        action = random.choice(legal_actions)
        move = env.decode_action(action)
        logger.info(f"[{self.name} - Random] Selected move: {move}")
        return action


class HumanPlayer(Player):
    """Prompts a human user for standard algebraic notation input (e.g., 'e2e4')."""
    def choose_action(self, env: PawnGameEnv) -> int:
        legal_moves = list(env.board.legal_moves)
        while True:
            try:
                user_input = input(f"[{self.name}] Enter move (e.g., e2e4): ").strip()
                move = chess.Move.from_uci(user_input)
                if chess.square_rank(move.to_square) in [0, 7]:
                    move.promotion = chess.QUEEN

                if move in legal_moves:
                    logger.info(f"[{self.name} - Human] Played valid move: {move}")
                    return move.from_square * 64 + move.to_square

                logger.warning(f"[{self.name} - Human] Input '{user_input}' is illegal right now.")
                print("Illegal move! Try again.")
            except ValueError:
                logger.warning(f"[{self.name} - Human] Failed parsing input format: '{user_input}'")
                print("Invalid format. Use notation like 'e2e4' or 'd7d5'.")


class SmartPlayer(Player):
    """Heuristic player optimizing promotion, captures, and progression."""
    def __init__(self, color, name):
        super().__init__(color, name)
        self.is_ai = True

    def choose_action(self, env: PawnGameEnv) -> int:
        legal_actions = env.get_legal_actions()
        best_action = legal_actions[0]
        best_score = -float('inf')

        for action in legal_actions:
            move = env.decode_action(action)
            score = 0

            # Feature 1: Immediate Promotion
            if chess.square_rank(move.to_square) == (7 if self.color == chess.WHITE else 0):
                score += 1000 

            # Feature 2: Captures
            if env.board.is_capture(move):
                score += 50

            # Feature 3: Prefer advancing pawns closer to the goal line
            distance_advanced = chess.square_rank(move.to_square)
            if self.color == chess.BLACK:
                distance_advanced = 7 - distance_advanced
            score += distance_advanced

            if score > best_score:
                best_score = score
                best_action = action

        chosen_move = env.decode_action(best_action)
        logger.info(f"[{self.name} - Smart] Evaluated {len(legal_actions)} moves. Picked {chosen_move} (Heuristic Score: {best_score})")
        return best_action


class RLAgentPlayer(Player):
    """An AI player that relies on a trained MaskablePPO neural network model."""
    def __init__(self, color: chess.Color, name: str, model_path: str):
        super().__init__(color, name)
        # Load the trained Stable-Baselines3 model parameters
        self.model = MaskablePPO.load(model_path)
        self.is_ai = True
        logger.info(f"[{self.name} - RL Agent] Successfully loaded model from '{model_path}'")

    def choose_action(self, env: PawnGameEnv) -> int:
        obs = env._get_obs()
        action_masks = env.action_masks()

        # The model predicts the best action based on the board state
        # We pass the action mask so it strictly evaluates legal moves
        action, _ = self.model.predict(obs, action_masks=action_masks, deterministic=True)

        chosen_move = env.decode_action(int(action))
        logger.info(f"[{self.name} - RL Agent] Brain selected move: {chosen_move}")
        return int(action)
