import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
    VecVideoRecorder,
)


ENV_ID = "MontezumaRevengeNoFrameskip-v4"


def make_single_env(seed: int, rank: int):
    def _init():
        env = gym.make(ENV_ID)
        env.reset(seed=seed + rank)
        env = AtariWrapper(env, terminal_on_life_loss=False)
        env = TimeLimit(env, max_episode_steps=4500)
        return env
    return _init


def make_env(n_envs: int = 16, seed: int = 0):
    env = DummyVecEnv([make_single_env(seed, i) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)
    return env


def make_video_env(step_tag: str):
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=0,
        env_kwargs={"render_mode": "rgb_array"},
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecVideoRecorder(
        env,
        video_folder="./videos/montezuma",
        record_video_trigger=lambda step: step == 0,
        video_length=4000,
        name_prefix=step_tag,
    )
    return env