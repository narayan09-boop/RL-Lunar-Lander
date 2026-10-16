"""Quick headless test to verify everything works."""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from reward_wrappers import RewardShapingWrapper, BrokenRewardWrapper, FixedRewardWrapper

# Test 1: Basic environment
print("Test 1: LunarLander-v3 environment...")
env = gym.make("LunarLander-v3")
obs, info = env.reset(seed=42)
print(f"  Observation shape: {obs.shape}")
print(f"  Observation: {obs}")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
print("  PASSED")
env.close()

# Test 2: PPO model creation
print("\nTest 2: PPO model creation...")
vec_env = make_vec_env("LunarLander-v3", n_envs=2, seed=42)
model = PPO("MlpPolicy", vec_env, verbose=0)
print("  Model created successfully")
# Quick tiny train
model.learn(total_timesteps=256)
print("  Short training completed")
vec_env.close()
print("  PASSED")

# Test 3: Reward wrappers
print("\nTest 3: Reward wrappers...")
for WrapperClass, name in [
    (RewardShapingWrapper, "RewardShapingWrapper"),
    (BrokenRewardWrapper, "BrokenRewardWrapper"),
    (FixedRewardWrapper, "FixedRewardWrapper"),
]:
    env = gym.make("LunarLander-v3")
    wrapped = WrapperClass(env)
    obs, info = wrapped.reset(seed=42)
    for _ in range(10):
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)
        if terminated or truncated:
            obs, info = wrapped.reset()
    wrapped.close()
    print(f"  {name}: PASSED")

# Test 4: RecordVideo wrapper
print("\nTest 4: RecordVideo wrapper...")
import os
os.makedirs("test_recordings", exist_ok=True)
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="test_recordings", episode_trigger=lambda x: True, name_prefix="test")
obs, info = env.reset(seed=42)
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
# Clean up test recordings
import shutil
shutil.rmtree("test_recordings", ignore_errors=True)
print("  PASSED")

print("\n" + "=" * 50)
print("ALL TESTS PASSED! Project is ready to use.")
print("=" * 50)
print("\nNext steps:")
print("  1. python 1_watch_random.py   (watch random agent crash)")
print("  2. python 2_train.py          (train PPO for 1M steps)")
print("  3. python 3_watch_trained.py  (watch it land)")
print("  4. python 4_train_shaped.py   (custom reward)")
print("  5. python 5_train_broken_and_fixed.py  (break & fix)")
print("  6. python 6_compare.py        (compare all)")
