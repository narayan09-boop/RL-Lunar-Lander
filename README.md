# Training an AI to Land a Rocket with Reward Shaping

## The Quick Version

Lunar Lander is a free game where you try to land a small rocket on a platform without crashing. It was originally part of OpenAI's Gym and is now available through Gymnasium, an open source library.

The idea is simple: instead of controlling the rocket yourself, you train an AI to do it. The AI tries landing thousands of times, getting rewarded when it does something right and penalized when it makes mistakes. Eventually, it figures out how to land on its own.

This is called **reinforcement learning**. It's basically like training a dog with treats, except the treats are points.

The AI learns based on the points it earns, so what you reward it for matters. By changing how points are awarded, you can guide it toward the behavior you want. This is called **reward shaping**, and it's what you'll be experimenting with in this project.

The same general approach is used in real robotics. Tesla has said its Optimus robot learned to dance entirely through reinforcement learning in simulation.

## How It Works

### 1. The game hands the AI eight numbers

The game (the **environment**) runs at 50 frames a second (a frame is one tick of the physics) and hands the AI eight numbers each frame, the **observation**. It never sees the picture, only these:

| # | What it is | Example |
|---|-----------|---------|
| 1 | left / right of the pad | 0.20 (a bit to the right) |
| 2 | height above the pad | 0.90 (still high) |
| 3 | sideways speed | -0.10 (drifting left) |
| 4 | up / down speed | 0.50 (falling) |
| 5 | tilt (in radians) | 0.05 (nearly upright) |
| 6 | how fast it is spinning | 0.00 (not spinning) |
| 7 | left leg on the ground | 0 (no) |
| 8 | right leg on the ground | 0 (no) |

### 2. The AI picks one of four actions

Do nothing, fire the left side engine, fire the main engine, or fire the right side engine. One choice per frame.

### 3. The brain is a function with dials

A **neural network** here is just a function: eight numbers go in, a preference for each action comes out. In between sit a few thousand dials (weights, random at the start) so the rocket flails. You don't build it: **Stable-Baselines3**, a free Python library, hands you one, small enough (two layers of 64 units) for a laptop.

### 4. Every frame gets a score

This is the **reward**, and Lunar Lander already comes with one. Each frame it works out a "how good is the position" number: minus 100 times the distance from the pad, minus 100 times the speed, minus 100 times the tilt, plus 10 per leg on the ground. The frame's reward is how much that number improved since the last frame, minus 0.3 if the main engine fired or 0.03 if a side engine did (fuel). Counting it's what you'll be experimenting with.

| Frame | Distance | Speed | Tilt | Legs down | Position score |
|-------|----------|-------|------|-----------|---------------|
| 1 | 0.50 | 0.40 | 0.10 | 0 | -50 - 40 - 10 = -100 |
| 2 | 0.49 | 0.38 | 0.10 | 0 | -49 - 38 - 10 = -97 |

The score improved by 3. The main engine fired to do it, so the frame pays 3 minus 0.3, which is 2.7. On the last frame the reward is replaced by plus 100 for resting safely or minus 100 for a crash. Add every frame up and you get the score for the attempt (an episode).

### 5. Training is play, score, adjust

The AI flies for a while, sees which choices led to more points, and nudges the dials toward them. Then it flies again. The nudging recipe is **PPO** (Proximal Policy Optimization, from OpenAI in 2017). Its one idea is to keep each update small, so one lucky stretch can't wreck what the AI has learned.

### 6. It learns exactly what you scored

An attempt is cut off at 1,000 frames, so a rocket that just hangs in the air can collect up to 1,000 points from your rule and pay at most 300 in fuel. Landing ends the attempt, and the payments stop, so hovering beats landing, and training can learn exactly that. This is called **reward hacking**. The fix is in the reward: say for getting better (the change between frames, like the built-in score does) instead of for sitting in a nice spot, and add a small fee every frame so waiting costs something.

## How You Would Make One

- Install Python, Gymnasium (with its Box2D physics add on) and Stable-Baselines3.
- Load Lunar Lander and watch a random rocket crash, then train PPO on it until it lands.
- Write your own reward rules on top, break it on purpose, then fix it, and record both.

Free, on a laptop, no graphics card needed. You need basic Python (loops, functions, what a class is). Budget an afternoon for a first landing and the rest of the day for the reward experiments. Keep the landing clip, your reward code and the hover clip and you have a resume project.

## If You Actually Want to Build It

### 1. Set Up (under an hour)

Install Python 3.10 or newer and make a **virtual environment** (a private folder of packages for this one project). In it, use pip (Python's installer, run in a terminal) to install Gymnasium with its "box2d" extra (Box2D is the lander's physics engine) and Stable-Baselines3, which brings PyTorch (the number-crunching library). It worked when Python imports all three without an error.

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
pip install gymnasium[box2d] stable-baselines3[extra]
```

### 2. Watch It Fail (20 minutes)

Create the environment by its ID, LunarLander-v3, with the game window on, and send it random actions in a loop (reset to start, step to act, each step gives back the new observation, the reward and whether the attempt ended). You should see a rocket tumble and crash, then restart. Print the eight numbers and match them to the table above.

### 3. Train It (half an hour or so of waiting)

Hand the environment to PPO with the standard small network (called MlpPolicy) and say how many steps to learn for (one step is one frame). The Hugging Face course says a million steps takes about 20 minutes on its Colab notebook (Google's free online notebook). Time yours. While it runs it prints ep_rew_mean, the average score per attempt over the last 100. That's the number to watch, and it should climb. Save the model when it finishes.

### 4. Watch It Land and Record It (30 minutes)

Load the saved model and let it fly. It should settle between the flags most of the time. To record, use Gymnasium's RecordVideo (it needs "rgb_array" mode and the moviepy package) or the GIF example on the Stable-Baselines3 examples page.

### 5. Write Your Own Reward (the real project, a few hours)

A **wrapper** is a thin layer around the environment that can change what comes out of it without touching its code. Yours sits between the game and PPO: each step it takes the built-in reward, adds or subtracts your own terms, and passes the total on. Start with a tiny charge per frame for taking too long (the built-in reward has none) and retrain. It should still land.

### 6. Break It, Then Fix It

Add the "plus 1 per frame" rule from above and retrain. Watch the clip, not just the score: the score can go up while the rocket hovers. It may take a retrain or two to show up. Then apply the fix, retrain, and keep both clips.

## How You Know It Works

One number: the average score over many attempts. Lunar Lander counts an attempt of 200 points or more as solved, so you want the average above that. Test with your wrapper off, over at least 10 attempts, so your own bonus can't flatter it. The curve starts well below zero and wobbles on the way up, which is normal.

## Where People Mess Up

- The Box2D install fails with an error that mentions "swig". Install swig first, then the box2d extra again.
- Old tutorials say LunarLander-v2. Current Gymnasium only has v3. Change the 2 to a 3.
- Trusting one run. Training is random, so run it more than once before you call a reward good or bad.

## Start Here

In this order:

1. [Stable-Baselines3: Install](https://stable-baselines3.readthedocs.io/en/master/guide/install.html). The exact install line.
2. [Gymnasium: Basic Usage](https://gymnasium.farama.org/introduction/basic_usage/). The reset and step loop (on a different game).
3. [Gymnasium: Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/). The eight numbers, the actions and the reward.
4. [Hugging Face Deep RL Course, Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction). A free unit that trains PPO on this game. (It says v2. Use v3.)
5. [Stable-Baselines3: Examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html). Training, saving, loading, GIFs.
6. [Gymnasium: Implementing Custom Wrappers](https://gymnasium.farama.org/introduction/create_custom_env/). How to write the wrapper for your reward rules (use the plain Wrapper section).
