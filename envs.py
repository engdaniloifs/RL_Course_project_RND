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
import pickle

ENV_ID = "MontezumaRevengeNoFrameskip-v4"




class EpisodeRecorderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_start_state = None

    def _clone_full_state(self):
        # include_rng=True is what you want for exact replay
        return self.unwrapped.clone_state(include_rng=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_start_state = self._clone_full_state()

        info = dict(info)
        info["episode_start_state"] = self.episode_start_state
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_actions.append(int(action))
        self.episode_rewards.append(float(reward))

        if terminated or truncated:
            info = dict(info)
            info["episode_actions"] = self.episode_actions.copy()
            info["episode_rewards"] = self.episode_rewards.copy()
            info["episode_start_state"] = self.episode_start_state

        return obs, reward, terminated, truncated, info


class MontezumaRoomWrapper(gym.Wrapper):
    def __init__(self, env, freeze_skull=True):
        super().__init__(env)
        self.freeze_skull = freeze_skull
        self.fixed_skull_pos = None  # will store (x, y) once

    def _get_room(self):
        ram = self.unwrapped.ale.getRAM()
        return int(ram[3])
    

    def _freeze_skull(self):
        self.unwrapped.ale.setRAM(47, 56)
        self.unwrapped.ale.setRAM(40, 93)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._freeze_skull()

        info["room"] = self._get_room()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # if for some reason it was not set on reset, set it here once
            

        self._freeze_skull()

        info["room"] = self._get_room()
        return obs, reward, terminated, truncated, info



class EpisodeSeedInfoWrapper(gym.Wrapper):
    def __init__(self, env, base_seed: int, env_rank: int):
        super().__init__(env)
        self.base_seed = int(base_seed)
        self.env_rank = int(env_rank)

        self.episode_idx = -1
        self.current_episode_seed = None

    def _compute_episode_seed(self) -> int:
        # Deterministic, unique per env and per episode
        # Large offset avoids collisions across workers
        return self.base_seed + self.env_rank * 1_000_000 + self.episode_idx

    def reset(self, *, seed=None, options=None):
        # Ignore external per-reset seed here so the schedule stays reproducible.
        # If you want, you can allow seed override, but this is simpler.
        self.episode_idx += 1
        self.current_episode_seed = self._compute_episode_seed()

        obs, info = self.env.reset(seed=self.current_episode_seed, options=options)

        info = dict(info)
        info["episode_seed"] = self.current_episode_seed
        info["episode_idx"] = self.episode_idx
        info["env_rank"] = self.env_rank
        info["base_seed"] = self.base_seed

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info = dict(info)

        # Only attach these on episode end if you want the callback to read them there.
        # You can also attach them every step; both are fine.
        if terminated or truncated:
            info["episode_seed"] = self.current_episode_seed
            info["episode_idx"] = self.episode_idx
            info["env_rank"] = self.env_rank
            info["base_seed"] = self.base_seed

        return obs, reward, terminated, truncated, info


def make_single_env(seed: int, rank: int):
    def _init():
        env = gym.make(ENV_ID,full_action_space=False)
        
        allowed_actions = [0, 1, 2, 3, 4, 5, 11, 12]
        env = TransformAction(
            env,
            func=lambda a: allowed_actions[a],
            action_space=Discrete(len(allowed_actions))
        )
        env = AtariWrapper(env,noop_max=0, terminal_on_life_loss=False, clip_reward=False)
        env = TimeLimit(env, max_episode_steps=4500)
        env = MontezumaRoomWrapper(env, freeze_skull=True)
        env = EpisodeSeedInfoWrapper(env, base_seed=seed, env_rank=rank)
        env = EpisodeRecorderWrapper(env)
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