import gymnasium as gym
from gymnasium.wrappers import TimeLimit,TransformAction
from gymnasium.spaces import Discrete
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
    VecVideoRecorder,
    SubprocVecEnv,
)


ENV_ID = "MontezumaRevengeNoFrameskip-v4"


class MontezumaRoomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _get_room(self):
        # Common Montezuma room byte in RAM
        ram = self.unwrapped.ale.getRAM()
        room = int(ram[3])
        return room

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["room"] = self._get_room()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["room"] = self._get_room()
        return obs, reward, terminated, truncated, info


def make_single_env(seed: int, rank: int):
    def _init():
        env = gym.make(ENV_ID,full_action_space=False, render_mode="rgb_array")
        
        allowed_actions = [0, 1, 2, 3, 4, 5, 11, 12]
        env = TransformAction(
            env,
            func=lambda a: allowed_actions[a],
            action_space=Discrete(len(allowed_actions))
        )
        env = AtariWrapper(env,noop_max=0, terminal_on_life_loss=False, clip_reward=False)
        env = TimeLimit(env, max_episode_steps=4500)
        env = MontezumaRoomWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_env(n_envs: int = 16, seed: int = 0):
    env = SubprocVecEnv([make_single_env(seed, i) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)
    #print observation space and action space
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    return env


def make_video_env(step_tag: str):
    env = DummyVecEnv([make_single_env(seed=0, rank=0)])
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