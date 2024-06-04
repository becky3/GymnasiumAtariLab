import os
import torch
import time

from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO

print(torch.version.cuda)
print(torch.cuda.is_available())


# Configurations
MODEL_NAME = "./models/a2c_breakout_04"
ENV_ID = "BreakoutNoFrameskip-v4"
N_ENVS = 8
SEED = 0
N_STACK = 4
VERBOSE = 1
DEVICE = "cuda"
BATCH_SIZE = 2048
TOTAL_TIMESTEPS = 50_000_000 - 250_000 - 12_000_000 - 10_000_000
# 250_000 (6/2)
# 12_000_000 (6/4)
# 10_000_000 (6/5) ログ確認そびれたのでおおよその値
# 学習済みSTEPS : 3500万
PROGRESS_BAR = True
EVAL_INTERVAL = 100000
SAVE_INTERVAL = 500000


def train():

    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    env = VecFrameStack(env, n_stack=N_STACK)

    model = PPO("CnnPolicy", env, verbose=VERBOSE, device=DEVICE, batch_size=BATCH_SIZE)

    start_time = time.time()
    total_loops = TOTAL_TIMESTEPS // EVAL_INTERVAL

    for i in range(0, TOTAL_TIMESTEPS, EVAL_INTERVAL):
        interval_start_time = time.time()

        print("=====================================")

        current_loop = i // EVAL_INTERVAL
        print(f"Step {i} to {i + EVAL_INTERVAL} (Loop {current_loop} of {total_loops})")

        print("learning...")
        model.learn(total_timesteps=EVAL_INTERVAL, progress_bar=PROGRESS_BAR)

        print("evaluating...")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )

        if i != 0 and i % SAVE_INTERVAL == 0:
            print("saving...")
            model.save(MODEL_NAME)
            print(f"******** Model saved at step {i}")

        interval_end_time = time.time()
        interval_time = interval_end_time - interval_start_time
        elapsed_time = interval_end_time - start_time
        remaining_time = (TOTAL_TIMESTEPS - i) * (elapsed_time / i) if i != 0 else 0
        remaining_time_hours = remaining_time / 3600

        print(f"After {i} steps, mean_reward={mean_reward:.2f} +/- {std_reward}")
        print(f"Interval time: {interval_time:.2f} sec")
        print(
            f"Estimated remaining time: {remaining_time_hours:.2f} H ({remaining_time:.2f} sec)"
        )

    model.save(MODEL_NAME)


def play_and_record():

    env = make_atari_env(ENV_ID, n_envs=1)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(MODEL_NAME, env=env)

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    video_folder = os.path.join("./videos", dt_string)

    env = VecVideoRecorder(
        env,
        video_folder=video_folder,
        record_video_trigger=lambda x: x == 2,
        video_length=10000000,
    )

    obs = env.reset()
    game_over = False
    total_reward = 0
    while not game_over:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards
        print(f"Observations: {obs}")
        print(f"Rewards: {rewards}")
        print(f"Dones: {dones}")
        print(f"Info: {info}")
        print(f"Total Reward: {total_reward}")
        game_over = info[0]["lives"] == 0
        env.render("human")

    env.close()


# train()
play_and_record()
