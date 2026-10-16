"""
Step 5: Custom Reward Wrapper
=============================
A **wrapper** is a thin layer around the environment that can change what comes
out of it without touching its code. This one sits between the game and PPO:
each step it takes the built-in reward, adds or subtracts your own terms, and
passes the total on.

This file contains:
    - RewardShapingWrapper: adds a tiny time penalty per frame (so hovering costs)
    - BrokenRewardWrapper:  adds +1 per frame (reward hacking — the rocket hovers)

Import and use these in the training scripts (4 and 5).
"""

import gymnasium as gym
import numpy as np


class RewardShapingWrapper(gym.Wrapper):
    """
    Custom reward wrapper that adds a small time penalty per frame.

    The built-in LunarLander reward has no cost for just existing, so a rocket
    can hover indefinitely and rack up position-improvement points. Adding a
    tiny charge per frame (default -0.1) discourages dawdling and encourages
    the AI to land quickly.

    You can add more reward terms here — this is where the real project lives.
    """

    def __init__(self, env, time_penalty=-0.1):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Store the original reward for debugging
        info["original_reward"] = reward

        # Add the time penalty: small cost each frame for not having landed yet
        shaped_reward = reward + self.time_penalty

        info["shaped_reward"] = shaped_reward
        info["time_penalty"] = self.time_penalty
        info["step_count"] = self.step_count

        return observation, shaped_reward, terminated, truncated, info


class BrokenRewardWrapper(gym.Wrapper):
    """
    DELIBERATELY BROKEN reward wrapper for demonstration purposes.

    This adds +1 per frame, which rewards the rocket for staying alive as long
    as possible. The AI learns to hover instead of land — a textbook example
    of **reward hacking**.

    At 1,000 frames max per episode, hovering collects up to +1,000 bonus on
    top of whatever position score it earns, which dwarfs the +100 landing
    bonus. So the AI never lands.

    The fix: use the change in position score between frames (like the built-in
    reward does) instead of a flat bonus, and keep the time penalty so waiting
    costs something.
    """

    def __init__(self, env, bonus_per_frame=1.0):
        super().__init__(env)
        self.bonus_per_frame = bonus_per_frame
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return super().reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        info["original_reward"] = reward

        # BROKEN: reward the AI for just existing — encourages hovering
        broken_reward = reward + self.bonus_per_frame

        info["broken_reward"] = broken_reward
        info["bonus_per_frame"] = self.bonus_per_frame
        info["step_count"] = self.step_count

        return observation, broken_reward, terminated, truncated, info


class FixedRewardWrapper(gym.Wrapper):
    """
    The FIXED version: combines the time penalty with improvement-based shaping.

    Instead of rewarding a flat bonus per frame (which encourages hovering),
    this rewards the *change* in a custom position metric AND charges a small
    fee per frame so the AI wants to land quickly.
    """

    def __init__(self, env, time_penalty=-0.1):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.prev_shaping = None
        self.step_count = 0

    def reset(self, **kwargs):
        self.prev_shaping = None
        self.step_count = 0
        return super().reset(**kwargs)

    def _compute_shaping(self, obs):
        """Compute a custom position quality metric from observation."""
        x_pos = obs[0]       # horizontal position
        y_pos = obs[1]       # vertical position (height)
        x_vel = obs[2]       # horizontal velocity
        y_vel = obs[3]       # vertical velocity
        angle = obs[4]       # tilt
        ang_vel = obs[5]     # angular velocity
        left_leg = obs[6]    # left leg contact
        right_leg = obs[7]   # right leg contact

        # Reward being centered, slow, and upright
        # Penalize distance from pad center, speed, and tilt
        shaping = (
            - 50 * np.sqrt(x_pos**2 + y_pos**2)   # distance to pad
            - 50 * np.sqrt(x_vel**2 + y_vel**2)    # speed
            - 50 * abs(angle)                        # tilt
            + 20 * left_leg                          # legs on ground
            + 20 * right_leg
        )
        return shaping

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        info["original_reward"] = reward

        # Compute improvement-based shaping (reward getting better, not being good)
        current_shaping = self._compute_shaping(observation)
        shaping_reward = 0.0
        if self.prev_shaping is not None:
            shaping_reward = current_shaping - self.prev_shaping
        self.prev_shaping = current_shaping

        # The final reward: original + improvement shaping + time penalty
        fixed_reward = reward + 0.5 * shaping_reward + self.time_penalty

        info["fixed_reward"] = fixed_reward
        info["shaping_reward"] = shaping_reward
        info["step_count"] = self.step_count

        return observation, fixed_reward, terminated, truncated, info
