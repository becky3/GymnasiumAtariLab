import gymnasium as gym
from gymnasium.utils.play import play

game_id = "BreakoutNoFrameskip-v4"
play(gym.make(game_id, render_mode="rgb_array"), zoom=4, fps=60)
