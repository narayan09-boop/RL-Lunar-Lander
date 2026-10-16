"""
Step 6: Break It, Then Fix It
=============================
Train with the BROKEN reward (+1 per frame) to demonstrate reward hacking —
the rocket learns to hover instead of land because staying alive earns more
than landing.

Then train with the FIXED reward and compare.

Saves both models so you can record clips of each.

Run:
    python 5_train_broken_and_fixed.py
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from reward_wrappers import BrokenRewardWrapper, FixedRewardWrapper


def make_broken_env(bonus=1.0):
    """Factory: LunarLander with broken reward (encourages hovering)."""
    def _init():
        env = gym.make("LunarLander-v3")
        env = BrokenRewardWrapper(env, bonus_per_frame=bonus)
        return env
    return _init


def make_fixed_env(time_penalty=-0.1):
    """Factory: LunarLander with fixed reward (improvement-based shaping)."""
    def _init():
        env = gym.make("LunarLander-v3")
        env = FixedRewardWrapper(env, time_penalty=time_penalty)
        return env
    return _init


def train_model(env, eval_env, model_name, total_timesteps=1_000_000):
    """Train a PPO model and save it."""
    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(f"logs/{model_name}", exist_ok=True)

    n_envs = 4

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}/",
        log_path=f"./logs/{model_name}/",
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
        tensorboard_log=f"./logs/tensorboard/",
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    model.save(f"models/ppo_{model_name}")
    print(f"\nModel saved to models/ppo_{model_name}.zip")

    return model


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    n_envs = 4

    # --- BROKEN: +1 per frame (reward hacking) ---
    print("=" * 60)
    print("PART 1: Training with BROKEN reward (+1 per frame)")
    print("The rocket should learn to HOVER instead of land.")
    print("=" * 60)

    broken_env = make_vec_env(make_broken_env(bonus=1.0), n_envs=n_envs)
    # Evaluate on the RAW environment — the real score
    eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=123)

    train_model(broken_env, eval_env, "broken", total_timesteps=1_000_000)

    broken_env.close()
    eval_env.close()

    # --- FIXED: improvement-based shaping + time penalty ---
    print("\n" + "=" * 60)
    print("PART 2: Training with FIXED reward (improvement + time penalty)")
    print("The rocket should learn to LAND properly.")
    print("=" * 60)

    fixed_env = make_vec_env(make_fixed_env(time_penalty=-0.1), n_envs=n_envs)
    eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=123)

    train_model(fixed_env, eval_env, "fixed", total_timesteps=1_000_000)

    fixed_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("DONE. Now run 6_compare.py to watch and record both.")
    print("=" * 60)


if __name__ == "__main__":
    main()
