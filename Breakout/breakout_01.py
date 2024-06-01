import os
import torch

from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO

print(torch.version.cuda)
print(torch.cuda.is_available())

model_name = "./Breakout/a2c_breakout_01"
# env_id = "BreakoutNoFrameskip-v0"
env_id = "BreakoutNoFrameskip-v4"
# env_id = "ALE/Breakout-v5"


def train():

    env = make_atari_env(env_id, n_envs=8, seed=0)
    env = VecFrameStack(env, n_stack=4)

    # env.metadata["render_fps"] = 60

    model = PPO("CnnPolicy", env, verbose=1, device="cuda")
    model.learn(total_timesteps=10_000_000)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    model.save(model_name)


def play_and_record():

    env = make_atari_env(env_id, n_envs=1)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_name, env=env)

    # env.metadata["render_fps"] = 60

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    video_folder = os.path.join("./videos", dt_string)

    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x % 1000 == 0,
    )

    obs = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")

    env.close()


train()
play_and_record()
