"""
Step 5: Train with Custom Reward Shaping
=========================================
Train PPO on LunarLander-v3 using the custom RewardShapingWrapper, which adds
a small time penalty per frame to discourage hovering.

The model is saved to ./models/ppo_shaped.zip when training finishes.

Run:
    python 4_train_shaped.py
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from reward_wrappers import RewardShapingWrapper


def make_shaped_env(time_penalty=-0.1):
    """Factory function that creates a LunarLander with reward shaping."""
    def _init():
        env = gym.make("LunarLander-v3")
        env = RewardShapingWrapper(env, time_penalty=time_penalty)
        return env
    return _init


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Create vectorized training environments with the reward wrapper
    n_envs = 4
    env = make_vec_env(make_shaped_env(time_penalty=-0.1), n_envs=n_envs)

    # Evaluation uses the UNWRAPPED environment so the score is comparable
    eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=123)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/shaped_best/",
        log_path="./logs/shaped/",
        eval_freq=50_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        tensorboard_log="./logs/tensorboard/",
    )

    print("=" * 60)
    print("Training PPO with REWARD SHAPING (time penalty = -0.1)")
    print("Evaluation uses the raw environment for fair comparison.")
    print("=" * 60)

    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save("models/ppo_shaped")
    print("\nShaped model saved to models/ppo_shaped.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
