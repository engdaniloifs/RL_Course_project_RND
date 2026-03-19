import json
import gymnasium as gym
import ale_py  # registers Atari envs

from gymnasium.wrappers import TimeLimit, TransformAction
from gymnasium.spaces import Discrete

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


ENV_ID = "MontezumaRevengeNoFrameskip-v4"


class MontezumaRoomWrapper(gym.Wrapper):
    def __init__(self, env, freeze_skull=True):
        super().__init__(env)
        self.freeze_skull = freeze_skull

    def _get_room(self):
        ram = self.unwrapped.ale.getRAM()
        return int(ram[3])

    def _freeze_skull(self):
        self.unwrapped.ale.setRAM(47, 56)
        self.unwrapped.ale.setRAM(40, 93)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.freeze_skull:
            self._freeze_skull()

        info = dict(info)
        info["room"] = self._get_room()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.freeze_skull:
            self._freeze_skull()

        info = dict(info)
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
        return self.base_seed + self.env_rank * 1_000_000 + self.episode_idx

    def reset(self, *, seed=None, options=None):
        # If an explicit seed is passed, use it. Otherwise keep deterministic schedule.
        if seed is not None:
            self.current_episode_seed = int(seed)
            self.episode_idx += 1
        else:
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
        if terminated or truncated:
            info["episode_seed"] = self.current_episode_seed
            info["episode_idx"] = self.episode_idx
            info["env_rank"] = self.env_rank
            info["base_seed"] = self.base_seed

        return obs, reward, terminated, truncated, info


def make_custom_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(
            env_id,
            full_action_space=False,
            render_mode="human",
        )

        allowed_actions = [0, 1, 2, 3, 4, 5, 11, 12]
        env = TransformAction(
            env,
            func=lambda a: allowed_actions[a],
            action_space=Discrete(len(allowed_actions)),
        )

        env = AtariWrapper(
            env,
            noop_max=0,
            terminal_on_life_loss=False,
            clip_reward=False,
        )

        env = TimeLimit(env, max_episode_steps=4500)
        env = MontezumaRoomWrapper(env, freeze_skull=True)
        env = EpisodeSeedInfoWrapper(env, base_seed=seed, env_rank=0)

        return env

    return _init


def watch_trained(model_path: str, env_id: str, meta_path: str | None = None, seed: int = 0):
    venv = DummyVecEnv([make_custom_env(env_id, seed)])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)

    print(f"Observation space: {venv.observation_space}")
    print(f"Action space: {venv.action_space}")

    model = PPO.load(model_path)

    episode_seed = None
    if meta_path is not None:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        episode_seed = meta.get("episode_seed")
        print(f"Loaded metadata from: {meta_path}")
        print(f"Saved reward: {meta.get('reward')}")
        print(f"Saved episode length: {meta.get('length')}")
        print(f"Saved episode seed: {episode_seed}")

    if episode_seed is not None:
        venv.seed(int(episode_seed))
    else:
        venv.seed(int(seed))

    obs = venv.reset()

    print("Watching trained model...")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)

            if dones[0]:
                print("Episode finished.")
                if episode_seed is not None:
                    print(f"Resetting with saved episode seed: {episode_seed}")
                    venv.seed(int(episode_seed))
                else:
                    print(f"Resetting with default seed: {seed}")
                    venv.seed(int(seed))

                obs = venv.reset()

    finally:
        venv.close()


def main():
    watch_trained(
        model_path="checkpoints/best_model.zip",
        env_id=ENV_ID,
        meta_path="checkpoints/best_episode.json",
        seed=0,
    )


if __name__ == "__main__":
    main()