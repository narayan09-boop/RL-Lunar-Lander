"""
Step 6 (continued): Compare Broken vs Fixed
============================================
Load the broken model (hovering) and the fixed model (landing), run them
side by side, and record video clips of each. This produces the two clips
you need for your project: the hover clip and the landing clip.

Run:
    python 6_compare.py
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO


def evaluate_model(model, model_name, episodes=10, record=True):
    """Run a model for several episodes, optionally recording video."""

    # --- Live viewing ---
    print(f"\n--- {model_name}: Live viewing ---")
    env = gym.make("LunarLander-v3", render_mode="human")

    scores = []
    for ep in range(1, min(4, episodes + 1)):  # Show 3 live
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        scores.append(total_reward)
        print(f"  Episode {ep}: reward = {total_reward:.2f}, steps = {steps}")

    env.close()

    # --- Recording ---
    if record:
        output_dir = f"recordings/{model_name}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n--- {model_name}: Recording {episodes} episodes ---")
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=output_dir,
            episode_trigger=lambda ep_id: True,
            name_prefix=model_name,
        )

        for ep in range(1, episodes + 1):
            obs, info = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            scores.append(total_reward)
            print(f"  Recorded episode {ep}: reward = {total_reward:.2f}, steps = {steps}")

        env.close()
        print(f"  Videos saved to ./{output_dir}/")

    avg = sum(scores) / len(scores) if scores else 0
    print(f"\n  {model_name} average reward: {avg:.2f} (over {len(scores)} episodes)")
    return avg


def main():
    results = {}

    # --- Standard PPO (baseline) ---
    baseline_path = "models/ppo_lunar_lander"
    if os.path.exists(baseline_path + ".zip"):
        print("\n" + "=" * 60)
        print("BASELINE: Standard PPO (no reward shaping)")
        print("=" * 60)
        model = PPO.load(baseline_path)
        results["baseline"] = evaluate_model(model, "baseline", episodes=5)

    # --- Shaped reward ---
    shaped_path = "models/ppo_shaped"
    if os.path.exists(shaped_path + ".zip"):
        print("\n" + "=" * 60)
        print("SHAPED: PPO with time penalty reward shaping")
        print("=" * 60)
        model = PPO.load(shaped_path)
        results["shaped"] = evaluate_model(model, "shaped", episodes=5)

    # --- Broken reward (hovering) ---
    broken_path = "models/ppo_broken"
    if os.path.exists(broken_path + ".zip"):
        print("\n" + "=" * 60)
        print("BROKEN: PPO with +1 per frame (reward hacking)")
        print("Expect the rocket to HOVER instead of land!")
        print("=" * 60)
        model = PPO.load(broken_path)
        results["broken"] = evaluate_model(model, "broken_hover", episodes=5)

    # --- Fixed reward ---
    fixed_path = "models/ppo_fixed"
    if os.path.exists(fixed_path + ".zip"):
        print("\n" + "=" * 60)
        print("FIXED: PPO with improvement-based shaping + time penalty")
        print("Expect the rocket to LAND properly!")
        print("=" * 60)
        model = PPO.load(fixed_path)
        results["fixed"] = evaluate_model(model, "fixed_landing", episodes=5)

    # --- Summary ---
    if results:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<20} {'Avg Reward':>12}")
        print("-" * 35)
        for name, score in results.items():
            solved = "✓ SOLVED" if score >= 200 else "✗ not solved"
            print(f"{name:<20} {score:>12.2f}  {solved}")
        print(f"\n(Lunar Lander is 'solved' at an average reward ≥ 200)")
    else:
        print("\nNo trained models found! Run the training scripts first:")
        print("  python 2_train.py                   (baseline)")
        print("  python 4_train_shaped.py             (shaped)")
        print("  python 5_train_broken_and_fixed.py   (broken + fixed)")


if __name__ == "__main__":
    main()
