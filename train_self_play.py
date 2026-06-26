import logging
import chess
import numpy as np
from sb3_contrib import MaskablePPO
from environment import PawnGameEnv
from logger_config import setup_logging

logger = logging.getLogger("SelfPlayTrainer")

class SelfPlayEnv(PawnGameEnv):
    """
    An environment where the agent plays against a static/frozen version 
    of its current neural network brain.
    """
    def __init__(self):
        super().__init__()
        self.opponent_model = None

    def set_opponent_model(self, model_path):
        """Loads a frozen checkpoint of the model to act as the Black opponent."""
        self.opponent_model = MaskablePPO.load(model_path)
        logger.info(f"Loaded frozen opponent checkpoint from {model_path}")

    def step(self, action_idx):
        # 1. Agent (White) takes its turn
        obs, reward, terminated, truncated, info = super().step(action_idx)

        # 2. If the game hasn't ended and an opponent model exists, Black takes its turn
        if not terminated and self.opponent_model is not None:
            # Reconstruct the board state from Black's perspective
            opp_obs = self._get_obs()
            opp_mask = self.action_masks()

            # Predict the opponent's best move using the frozen neural net
            opp_action, _ = self.opponent_model.predict(opp_obs, action_masks=opp_mask, deterministic=False)

            # Execute opponent's move
            obs, opp_reward, terminated, truncated, info = super().step(int(opp_action))

            # If Black won, the training agent (White) receives a loss penalty
            if terminated and opp_reward == -10.0:
                reward = -10.0

        return obs, reward, terminated, truncated, info


def run_self_play_pipeline(iterations=5, steps_per_iteration=20000):
    setup_logging()

    env = SelfPlayEnv()

    # Iteration 0: Initialize a completely fresh baseline model
    logger.info("Initializing baseline model for iteration 0...")
    model = MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    # Save this initial baseline
    current_model_path = "self_play_agent_latest"
    model.save(current_model_path)

    for i in range(1, iterations + 1):
        logger.info(f"\n=== STARTING SELF-PLAY ITERATION {i}/{iterations} ===")

        # Freeze the current best model and hand it to the environment as the opponent
        env.set_opponent_model(current_model_path)

        # Reload the network weights to continue training against the copy
        model = MaskablePPO.load(current_model_path, env=env, tensorboard_log="./pawn_self_play_logs/")

        # Learn against yourself!
        logger.info(f"Training agent for {steps_per_iteration} timesteps...")
        model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)

        # Save the upgraded model over the old checkpoint
        model.save(current_model_path)
        logger.info(f"Iteration {i} complete. Model checkpoint updated.")

    logger.info("\nSelf-play loop fully complete! Final model saved as 'self_play_agent_latest.zip'")


if __name__ == "__main__":
    # Adjust iterations and timesteps based on your available hardware performance
    run_self_play_pipeline(iterations=5, steps_per_iteration=30000)
