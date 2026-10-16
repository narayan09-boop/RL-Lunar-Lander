"""
Step 2: Watch It Fail
=====================
Create the LunarLander-v3 environment with rendering on, and send it random
actions so you can see a rocket tumble and crash. Prints the eight observation
numbers each frame so you can match them to the table in the README.

Run:
    python 1_watch_random.py
"""

import gymnasium as gym


def main():
    # Create the environment with human-visible rendering
    env = gym.make("LunarLander-v3", render_mode="human")

    observation, info = env.reset(seed=42)

    # Column headers matching the eight observation numbers
    headers = [
        "x-pos",     # 1: left/right of the pad
        "y-pos",     # 2: height above the pad
        "x-vel",     # 3: sideways speed
        "y-vel",     # 4: up/down speed
        "angle",     # 5: tilt in radians
        "ang-vel",   # 6: how fast it is spinning
        "left-leg",  # 7: left leg on the ground (0 or 1)
        "right-leg", # 8: right leg on the ground (0 or 1)
    ]

    print(f"{'Frame':>6}  " + "  ".join(f"{h:>9}" for h in headers) + "   Reward  Done")
    print("-" * 120)

    total_reward = 0
    episode = 1
    frame = 0

    for step in range(2000):
        # Pick a random action: 0=nothing, 1=left engine, 2=main engine, 3=right engine
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        frame += 1

        # Print the eight numbers, the reward, and whether the episode ended
        obs_str = "  ".join(f"{v:9.4f}" for v in observation)
        done = terminated or truncated
        print(f"{frame:6d}  {obs_str}  {reward:7.2f}  {done}")

        if done:
            print(f"\n--- Episode {episode} finished | Total reward: {total_reward:.2f} ---\n")
            observation, info = env.reset()
            total_reward = 0
            frame = 0
            episode += 1

            if episode > 5:
                break

    env.close()
    print("Done. You should have seen the rocket tumble and crash several times.")


if __name__ == "__main__":
    main()
