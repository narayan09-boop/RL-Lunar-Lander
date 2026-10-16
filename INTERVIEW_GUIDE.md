# 🎯 RL Lunar Lander & Reward Shaping: Interview Master Guide

This guide is structured to help you ace technical and conceptual questions in an interview, transitioning smoothly from high-level summaries to mathematical and architectural deep dives.

---

## 1. The 30-Second Elevator Pitch

> *"In this project, I used Reinforcement Learning to autonomously land a rocket in Gymnasium's LunarLander-v3 environment using PPO from Stable-Baselines3. The core focus wasn't just solving the game—which requires an average return of ≥ 200 points—but experimenting with **reward shaping** and analyzing **reward hacking**. I demonstrated how a naive reward bonus (+1 per frame) caused the agent to exploit the environment by hovering endlessly to harvest points rather than landing. I then resolved this using improvement-based potential shaping and per-step time penalties, achieving efficient and stable landings without unintended behavioral side effects."*

---

## 2. Core Concepts You Must Know Cold

### A. The RL Formulation (MDP: Markov Decision Process)
* **Environment**: 2D physics simulation powered by Box2D (`LunarLander-v3`). Runs at 50 FPS (50 Hz).
* **Observation Space (Continuous, $\mathbb{R}^8$)**:
  1. Horizontal position $x$ (relative to pad center: $[-1, 1]$)
  2. Vertical position $y$ (height above pad)
  3. Linear velocity $v_x$ (horizontal drift)
  4. Linear velocity $v_y$ (vertical speed / falling rate)
  5. Angle $\theta$ (tilt in radians)
  6. Angular velocity $\omega$ (rotational speed)
  7. Left leg ground contact flag (boolean 0 or 1)
  8. Right leg ground contact flag (boolean 0 or 1)
* **Action Space (Discrete, 4 actions)**:
  * `0`: Do nothing
  * `1`: Fire left orientation engine
  * `2`: Fire main thruster
  * `3`: Fire right orientation engine
* **Episode Termination**:
  * Crashes or comes to a complete rest (Terminated).
  * Reaches timeout cut-off at 1,000 steps (Truncated).
* **Success Criteria**: An average evaluation score $\ge +200$ points across 100 consecutive episodes.

---

### B. The Algorithm: PPO (Proximal Policy Optimization)
**Q: Why PPO instead of DQN or Policy Gradient (REINFORCE)?**
* **DQN** works for discrete action spaces, but can suffer from instability with Q-value overestimation and doesn't handle continuous-like trajectory dynamics as smoothly as actor-critic methods.
* **Standard Policy Gradient (REINFORCE)** has high variance and is prone to destructive policy updates: one bad step with a huge gradient update can collapse policy performance irreversibly.
* **PPO** (Schulman et al., 2017) introduces a **clipped surrogate objective function**:
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the Generalized Advantage Estimator (GAE).
* **Why it matters**: The clipping parameter $\epsilon$ (typically $0.2$) penalizes overly large policy changes, providing the monotonic improvement guarantees of TRPO (Trust Region Policy Optimization) but using simple first-order stochastic gradient descent.

---

### C. Neural Network Architecture (`MlpPolicy`)
* **Type**: Multi-Layer Perceptron (MLP) Actor-Critic architecture.
* **Layers**: Two hidden layers with 64 units each, separated by Tanh activations.
* **Actor Head**: Takes state $\mathbb{R}^8 \to$ outputs logits for the 4 discrete actions (Categorical distribution).
* **Critic Head**: Takes state $\mathbb{R}^8 \to$ outputs state value estimate $V(s) \in \mathbb{R}^1$.
* **Footprint**: Extremely compact (~few thousand parameters), allowing full 1,000,000-step training in 20-30 minutes on a CPU.

---

## 3. The Highlight: Reward Shaping & Reward Hacking

### The Default Reward Mechanics
LunarLander's built-in reward already incorporates potential-based shaping:
1. Improvement in distance to landing pad: $-\sqrt{x^2 + y^2}$
2. Improvement in linear speed: $-\sqrt{v_x^2 + v_y^2}$
3. Improvement in tilt angle: $-|\theta|$
4. Leg contact: $+10$ points per grounded leg.
5. Fuel consumption costs: $-0.3$ per frame for main engine, $-0.03$ per frame for side thrusters.
6. Terminal outcome: $+100$ for a successful safe landing, $-100$ for a crash.

---

### The "Reward Hacking" Experiment
**Q: How did you break the environment?**
* **Change**: Added $+1.0$ reward for every frame the rocket remained active.
* **Intent**: To reward the lander for not crashing.
* **What the AI actually learned**:
  * An episode runs up to 1,000 steps max.
  * Sitting stationary/hovering at altitude yields:
    $$\text{Hovering Return} \approx 1000 \times (+1.0) - (\text{fuel cost } \sim 0.3 \times 1000) \approx +700 \text{ points}$$
  * A fast, clean landing cuts the episode short at ~200 steps:
    $$\text{Landing Return} \approx (200 \times 1.0) + 100 (\text{landing bonus}) - \text{fuel} \approx +250 \text{ points}$$
  * **Result**: The policy converged to hovering in place until the 1,000-step time limit was reached! The AI optimized for the metric, not the intent.

---

### The Fix: Potential-Based Reward Shaping & Time Penalties
**Q: How did you mathematically resolve the hovering loop?**
1. **Time Penalty (Step Cost)**: Implemented a small per-step penalty ($-0.1$/frame). Now, lingering costs points rather than generating them.
2. **Difference-based (Potential) Shaping**:
   According to Ng, Harada, and Russell (1999) (*"Policy invariance under reward transformations"*), reward shaping preserves optimal policy invariance if and only if the shaping term is a difference of potential functions:
   $$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
   Rather than awarding flat bonuses for *being* in a good state, the reward only triggers when *improving* state potential:
   $$\Delta \Phi = \Phi(s_{t}) - \Phi(s_{t-1})$$
   This prevents cyclical reward loops (infinite reward cycles) while maintaining incentives to land quickly.

---

## 4. Top Technical Interview Q&A

### Q1: "Why evaluate the agent without the wrapper enabled?"
> **Answer**:
> *"When shaping rewards, the scalar return includes synthetic penalties or bonuses we added to guide exploration. If we measured validation scores using the shaped reward, a high score could simply reflect our artificial bonuses rather than genuine task completion. By peeling off the wrapper and evaluating on raw Gymnasium `LunarLander-v3`, we obtain a standardized, unskewed metric: an average score $\ge 200$ on the unmodified game physics."*

---

### Q2: "What hyperparameters were most influential in your PPO run?"
> **Answer**:
> 1. **`n_steps` and `batch_size`**: Dictates how much experience is collected before updating policy weights. Vectorizing with 4 parallel environments (`n_envs=4`) smoothed gradient estimates and prevented policy collapse.
> 2. **`gamma` ($\gamma = 0.99$)**: The discount factor. High discount ensures long-horizon planning, crucial because the $+100$ landing payoff is delayed by hundreds of steps.
> 3. **`gae_lambda` ($\lambda = 0.95$)**: Trades off bias vs variance in advantage calculation.
> 4. **`ent_coef` (Entropy coefficient)**: Controls exploration. Prevents the actor policy from prematurely collapsing to deterministic actions in the early flailing phase.

---

### Q3: "What's the difference between `terminated` and `truncated` in Gymnasium?"
> **Answer**:
> *"In Gymnasium (v26+ and v3 environments):*
> * `terminated` reflects an endogenous Markov condition: the agent crashed or cleanly landed (absorbing terminal state). Here, the value of the next state $V(s') = 0$.
> * `truncated` indicates an exogenous constraint: the maximum step limit (1,000 frames) was reached. The episode ended, but the state wasn't physically terminal. For bootstrapping value updates, we must not treat $V(s')$ as zero, otherwise the agent artificially assumes timeout is death."*

---

### Q4: "How does this simulation map to real-world robotics (Sim2Real)?"
> **Answer**:
> *"In real robotics—such as SpaceX booster landing or robotic manipulation—training directly on physical hardware is hazardous and expensive. We train in physics simulators (like Box2D, Isaac Gym, or MuJoCo). The biggest challenge is the **Reality Gap** (Sim2Real transfer). To bridge this in production, practitioners employ **Domain Randomization** (varying wind gusts, mass, thruster lags, sensor noise) and train robust policies using PPO that generalize to real-world disturbances."*

---

## 5. Cheat Sheet of Quick Metrics

| Metric | Value |
|---|---|
| Environment ID | `LunarLander-v3` |
| Physics Engine | Box2D (2D rigid body physics) |
| FPS / Control Frequency | 50 Hz |
| Solved Threshold | Mean score $\ge 200$ across 100 evaluation episodes |
| Observations | 8 continuous floats |
| Actions | 4 discrete integers |
| Training Budget | 1,000,000 timesteps (~20-25 mins CPU) |
| Vectorized Workers | 4 parallel sub-environments |
