import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3 import A2C


print(torch.version.cuda)
print(torch.cuda.is_available())

model_name = "./Breakout/a2c_breakout_01"
env_id = "ALE/Breakout-v5"


def train():
    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    vec_env = make_atari_env(env_id, n_envs=8, seed=0)
    # Frame-stacking with 4 frames
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = A2C("CnnPolicy", vec_env, verbose=1, device="cuda")
    model.learn(total_timesteps=50_000)

    # Save the model
    model.save(model_name)


def play_and_record():

    # Create a single environment for gameplay
    env = make_atari_env(env_id, n_envs=1)
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    model = A2C.load(model_name, env=env)

    # Set the render_fps in the environment metadata
    env.metadata["render_fps"] = 60

    # Wrap the environment to record gameplay
    env = VecVideoRecorder(
        env,
        video_folder="./videos",
        record_video_trigger=lambda x: x % 1000 == 0,
    )

    obs = env.reset()
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render("human")

    env.close()


# train()
play_and_record()
