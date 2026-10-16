"""
Step 4: Watch It Land and Record It
====================================
Load the saved PPO model and let it fly. It should settle between the flags
most of the time. Records a video using Gymnasium's RecordVideo wrapper.

Run:
    python 3_watch_trained.py
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO


def watch_live(model, episodes=5):
    """Watch the trained agent land in a live window."""
    env = gym.make("LunarLander-v3", render_mode="human")

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # The model picks an action (deterministic = always pick the best)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep}: Total reward = {total_reward:.2f}")

    env.close()


def record_video(model, output_dir="recordings", episodes=5):
    """Record the trained agent landing as video files."""
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=output_dir,
        episode_trigger=lambda ep_id: True,  # Record every episode
        name_prefix="lunar_lander_trained",
    )

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Recorded episode {ep}: Total reward = {total_reward:.2f}")

    env.close()
    print(f"\nVideos saved to ./{output_dir}/")


def main():
    # Try to load the best model first, fall back to the final model
    model_path = "models/best_model"
    if not os.path.exists(model_path + ".zip"):
        model_path = "models/ppo_lunar_lander"

    if not os.path.exists(model_path + ".zip"):
        print("No trained model found! Run 2_train.py first.")
        return

    print(f"Loading model from {model_path}.zip")
    model = PPO.load(model_path)

    # Watch it live
    print("\n--- Watching the trained agent (live) ---")
    watch_live(model, episodes=3)

    # Record videos
    print("\n--- Recording videos ---")
    record_video(model, episodes=3)


if __name__ == "__main__":
    main()
