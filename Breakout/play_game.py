import time

import keyboard
import gymnasium as gym

env = gym.make(
    "ALE/Breakout-v5",
    render_mode="human",
    full_action_space=False,
    obs_type="rgb",
)
env.reset()

action_dict = {"w": 0, "s": 1, "d": 2, "a": 3}

terminated = False
total_reward = 0
while True:

    event = keyboard.read_event()
    print(event.name)

    reward = 0
    # if action_dict.get(event.name, -1) != -1:
    obs, reward, terminated, _, info = env.step(action_dict.get(event.name, -1))
    total_reward += reward
    env.render()

env.reset()
env.close()
