import logging
import chess
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from environment import PawnGameEnv
from players import SmartPlayer
from logger_config import setup_logging

logger = logging.getLogger("Trainer")

class RLTrainingEnv(PawnGameEnv):
    """
    A custom wrapper training environment.
    The RL Agent plays White. The SmartPlayer plays Black.
    """
    def __init__(self):
        super().__init__()
        # The bot our agent needs to learn how to defeat
        self.opponent = SmartPlayer(chess.BLACK, "TrainerBot")

    def step(self, action_idx):
        # 1. RL Agent (White) takes its turn
        obs, reward, terminated, truncated, info = super().step(action_idx)
        
        # 2. If the game didn't end, the Smart Bot (Black) takes its turn immediately
        if not terminated:
            opponent_action = self.opponent.choose_action(self)
            obs, opp_reward, terminated, truncated, info = super().step(opponent_action)
            
            # If the opponent won, give the agent a negative reward
            if terminated and opp_reward == -10.0:
                reward = -10.0
                
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    setup_logging()
    logger.info("Initializing Reinforcement Learning Training Environment...")
    
    env = RLTrainingEnv()
    
    # Configure Maskable PPO with a Multi-Layer Perceptron (Neural Network) Policy
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log="./pawn_rl_logs/"
    )
    
    logger.info("Starting training loop (50,000 timesteps)... This might take a few minutes.")
    model.learn(total_timesteps=50000)
    
    # Save the trained brain parameters
    model_name = "pawn_game_rl_agent"
    model.save(model_name)
    logger.info(f"Training complete! Model saved locally as '{model_name}.zip'")
    