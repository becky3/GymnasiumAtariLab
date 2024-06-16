import os
import torch
import time
import logging

from datetime import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import PPO

now = datetime.now()
dt_string = now.strftime("%Y%m%d_%H%M%S")

# Configurations
MODEL_NAME = "./models/a2c_breakout_04"
ENV_ID = "BreakoutNoFrameskip-v4"
LOG_NAME = f"./logs/breakout_04_{dt_string}.log"
N_ENVS = 8
SEED = 0
N_STACK = 4
VERBOSE = 1
DEVICE = "cuda"
BATCH_SIZE = 2048
TOTAL_TIMESTEPS = 10_000_000
# 10_000_000 (6/6)
PROGRESS_BAR = True
EVAL_INTERVAL = 100000
SAVE_INTERVAL = 500000


def setup_logger():
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("breakout.log")
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    format = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)


setup_logger()

logging.info(torch.version.cuda)
logging.info(torch.cuda.is_available())


def train():
    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    env = VecFrameStack(env, n_stack=N_STACK)

    # Check if model exists and load it, else create a new one
    if os.path.isfile(MODEL_NAME + ".zip"):
        logging.info("Loading existing model...")
        model = PPO.load(MODEL_NAME, env=env)
    else:
        logging.info("Creating new model...")
        model = PPO(
            "CnnPolicy", env, verbose=VERBOSE, device=DEVICE, batch_size=BATCH_SIZE
        )

    # Rest of the code...
    start_time = time.time()
    total_loops = TOTAL_TIMESTEPS // EVAL_INTERVAL

    for i in range(0, TOTAL_TIMESTEPS, EVAL_INTERVAL):
        interval_start_time = time.time()

        logging.info("=====================================")

        current_loop = i // EVAL_INTERVAL
        logging.info(
            f"Step {i} to {i + EVAL_INTERVAL} (Loop {current_loop} of {total_loops})"
        )

        logging.info("learning...")
        model.learn(total_timesteps=EVAL_INTERVAL, progress_bar=PROGRESS_BAR)

        logging.info("evaluating...")
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )

        if i % SAVE_INTERVAL == 0:
            logging.info("saving...")
            model.save(MODEL_NAME)
            logging.info(f"******** Model saved at step {i}")

        interval_end_time = time.time()
        interval_time = interval_end_time - interval_start_time
        elapsed_time = interval_end_time - start_time
        remaining_time = (TOTAL_TIMESTEPS - i) * (elapsed_time / i) if i != 0 else 0
        remaining_time_hours = remaining_time / 3600

        logging.info(f"After {i} steps, mean_reward={mean_reward:.2f} +/- {std_reward}")
        logging.info(f"Interval time: {interval_time:.2f} sec")
        logging.info(
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
        # logging.info(f"Observations: {obs}")
        # logging.info(f"Rewards: {rewards}")
        # logging.info(f"Dones: {dones}")
        # logging.info(f"Info: {info}")
        # logging.info(f"Total Reward: {total_reward}")
        game_over = info[0]["lives"] == 0
        env.render("human")

    env.close()


# train()
play_and_record()
