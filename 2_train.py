"""
Step 3: Train It
================
Hand the LunarLander-v3 environment to PPO with the standard small network
(MlpPolicy — two hidden layers of 64 units) and train for 1,000,000 steps.

While it runs it prints ep_rew_mean (the average score per attempt over the
last 100). That's the number to watch, and it should climb.

The model is saved to ./models/ppo_lunar_lander.zip when training finishes.

Run:
    python 2_train.py
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def main():
    # Create the training environment (no rendering during training — much faster)
    env = make_vec_env("LunarLander-v3", n_envs=4, seed=42)

    # Create a separate evaluation environment
    eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=123)

    # Create the models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Evaluate every 50k steps, save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=50_000 // 4,   # divided by n_envs
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Create the PPO agent with the standard small network (MlpPolicy)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=42,
        tensorboard_log="./logs/tensorboard/",
    )

    print("=" * 60)
    print("Training PPO on LunarLander-v3 for 1,000,000 steps...")
    print("Watch ep_rew_mean — it should climb over time.")
    print("=" * 60)

    # Train for 1,000,000 steps
    model.learn(
        total_timesteps=1_000_000,
        callback=eval_callback,
        progress_bar=True,
    )

    # Save the final model
    model.save("models/ppo_lunar_lander")
    print("\nModel saved to models/ppo_lunar_lander.zip")
    print("Best model (from evaluation) saved to models/best_model.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
