import logging
import chess
from sb3_contrib import MaskablePPO
from environment import PawnGameEnv
from logger_config import setup_logging

logger = logging.getLogger("SelfPlayTrainer")

class SelfPlayEnv(PawnGameEnv):
    def __init__(self):
        super().__init__()
        self.opponent_model = None

    def set_opponent_model(self, model_path):
        self.opponent_model = MaskablePPO.load(model_path)
        logger.info(f"Loaded frozen opponent checkpoint from {model_path}")

    def reset(self, seed=None, options=None):
        # Force the environment to scramble the opening position for diverse practice
        if options is None:
            options = {}
        options["scramble_board"] = True
        return super().reset(seed=seed, options=options)

    def old_step(self, action_idx):
        # 1. Agent (White) takes its turn
        obs, reward, terminated, truncated, info = super().step(action_idx)

        # 2. Black takes its turn using the frozen model
        if not terminated and self.opponent_model is not None:
            opp_obs = self._get_obs()
            opp_mask = self.action_masks()

            # Use deterministic=False for the opponent too, creating unpredictable games
            opp_action, _ = self.opponent_model.predict(opp_obs, action_masks=opp_mask, deterministic=False)

            obs, opp_reward, terminated, truncated, info = super().step(int(opp_action))

            if terminated and opp_reward == -10.0:
                reward = -10.0

        return obs, reward, terminated, truncated, info


    def step(self, action_idx):
        # 1. Training Agent (White) takes its turn
        obs, reward, terminated, truncated, info = super().step(action_idx)

        # 2. Opponent (Black) takes its turn using the frozen model
        if not terminated and self.opponent_model is not None:
            opp_obs = self._get_obs()
            opp_mask = self.action_masks()

            opp_action, _ = self.opponent_model.predict(opp_obs, action_masks=opp_mask, deterministic=False)

            # Use the environment's step directly to let material changes alter the reward
            obs, reward, terminated, truncated, info = super().step(int(opp_action))

        return obs, reward, terminated, truncated, info


def run_self_play_pipeline(iterations=10, steps_per_iteration=40000):
    setup_logging()
    env = SelfPlayEnv()

    # Iteration 0: High ent_coef forces maximum initial experimentation
    logger.info("Initializing high-exploration baseline model for iteration 0...")
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        ent_coef=0.08,    # Boosted exploration
        n_steps=2048
    )

    current_model_path = "self_play_agent_latest"
    model.save(current_model_path)

    for i in range(1, iterations + 1):
        logger.info(f"\n=== STARTING PRO SELF-PLAY ITERATION {i}/{iterations} ===")

        env.set_opponent_model(current_model_path)

        # Reload while keeping the exploration rules active
        model = MaskablePPO.load(
            current_model_path, 
            env=env, 
            tensorboard_log="./pawn_self_play_logs/",
            custom_objects={"ent_coef": max(0.01, 0.08 - (i * 0.007))} # Slowly decays exploration over time
        )

        logger.info(f"Training agent for {steps_per_iteration} timesteps...")
        model.learn(total_timesteps=steps_per_iteration, reset_num_timesteps=False)

        model.save(current_model_path)
        logger.info(f"Iteration {i} complete. Model checkpoint updated.")

    logger.info("\nSupercharged training complete! Final model saved as 'self_play_agent_latest.zip'")


if __name__ == "__main__":
    # Total of 400,000 steps across diverse openings. Let this cook!
    run_self_play_pipeline(iterations=10, steps_per_iteration=40000)
