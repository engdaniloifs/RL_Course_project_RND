import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rnd import RNDModel
from envs import make_video_env
from pathlib import Path
import json
from datetime import datetime


class RNDBonusCallback(BaseCallback):
    def __init__(self, rnd_model: RNDModel,intrinsic_coefficient, extrinsic_coefficient, verbose=0):
        super().__init__(verbose)
        self.rnd = rnd_model
        self.rollout_next_obs = []
        self.intrinsic_coefficient = intrinsic_coefficient
        self.extrinsic_coefficient = extrinsic_coefficient
        self.best_reward = float("-inf")
    def _on_rollout_start(self) -> None:
        self.rollout_next_obs = []

    def _on_step(self) -> bool:
        new_obs = self.locals["new_obs"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        step_next_obs = new_obs.copy()
        for idx, done in enumerate(dones):
            if done and infos[idx].get("terminal_observation") is not None:
                step_next_obs[idx] = infos[idx]["terminal_observation"]

        self.rollout_next_obs.append(step_next_obs)
        return True

    def _on_rollout_end(self) -> None:
        next_obs = np.asarray(self.rollout_next_obs)
        n_steps, n_envs = next_obs.shape[:2]

        flat_next_obs = next_obs.reshape(n_steps * n_envs, *next_obs.shape[2:])
        rnd_next_obs = flat_next_obs[:, -1:, :, :]
        obs_float = rnd_next_obs.astype(np.float32)

        norm_obs = self.rnd.normalize_obs(obs_float)
        chunk_size = 512
        for start in range(0, obs_float.shape[0], chunk_size):
            self.rnd.obs_rms.update(obs_float[start:start + chunk_size])

        tgt_full = self.rnd.compute_target_features(norm_obs)
        intrinsic = self.rnd.compute_intrinsic_reward(norm_obs, tgt=tgt_full)
        intrinsic = intrinsic.reshape(n_steps, n_envs)
        intrinsic_scaled = intrinsic * self.intrinsic_coefficient

        if len(self.model.ep_info_buffer) > 0:
            current_best = max(ep_info["r"] for ep_info in self.model.ep_info_buffer)

            if current_best > self.best_reward:
                self.best_reward = current_best

            self.logger.record("test/reward", float(self.best_reward))

        self.model.rollout_buffer.rewards = self.model.rollout_buffer.rewards * self.extrinsic_coefficient + intrinsic_scaled

        last_values = self.locals["values"]
        dones = self.locals["dones"]
        self.model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=dones,
        )

        rnd_loss = self.rnd.update(norm_obs, tgt_full=tgt_full)

        self.logger.record("rnd/loss", float(rnd_loss))
        self.logger.record("rnd/intrinsic_mean", float(intrinsic_scaled.mean()))
        self.logger.record("rnd/intrinsic_std", float(intrinsic_scaled.std()))
        self.logger.record("rnd/intrinsic_max", float(intrinsic_scaled.max()))


class RoomLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.global_rooms = set()
        self.local_rooms = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.local_rooms = [set() for _ in range(n_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            room = info.get("room")
            if room is not None:
                self.local_rooms[i].add(room)
                self.global_rooms.add(room)

        self.logger.record("rooms/total_unique_rooms", len(self.global_rooms))

        for i, done in enumerate(dones):
            if done:
                visited = self.local_rooms[i]

                self.logger.record("rooms/episode_num_rooms", len(visited))
                if visited:
                    self.logger.record("rooms/episode_max_room", max(visited))

                self.local_rooms[i].clear()

        return True


class BestPolicySaverCallback(BaseCallback):
    def __init__(self, save_path="checkpoints", verbose=0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.archive_path = self.save_path / "archive"
        self.archive_path.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.save_path / "best_episode.json"
        self.model_path = self.save_path / "best_model"

        self.best_episode_reward = float("-inf")
        self._load_previous_best()

    def _load_previous_best(self) -> None:
        if not self.meta_path.exists():
            return

        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.best_episode_reward = float(metadata.get("reward", float("-inf")))

            if self.verbose > 0:
                print(f"Loaded previous best reward: {self.best_episode_reward:.2f}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not load previous best metadata: {e}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for env_idx, info in enumerate(infos):
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            ep_reward = float(ep_info["r"])

            if ep_reward > self.best_episode_reward:
                old_best = self.best_episode_reward
                self.best_episode_reward = ep_reward

                # Save/update the main "best" checkpoint
                self.model.save(str(self.model_path))

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                reward_str = f"{ep_reward:.2f}".replace(".", "p")
                archive_model_path = self.archive_path / f"best_model_reward_{reward_str}_step_{self.num_timesteps}_{timestamp}"
                self.model.save(str(archive_model_path))

                metadata = {
                    "reward": ep_reward,
                    "length": int(ep_info["l"]),
                    "num_timesteps": int(self.num_timesteps),
                    "env_idx": int(env_idx),
                    "episode_seed": info.get("episode_seed"),
                    "saved_at": timestamp,
                    "main_model_path": str(self.model_path.with_suffix(".zip")),
                    "archive_model_path": str(archive_model_path.with_suffix(".zip")),
                }

                with open(self.meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                archive_meta_path = self.archive_path / f"best_episode_reward_{reward_str}_step_{self.num_timesteps}_{timestamp}.json"
                with open(archive_meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                if self.verbose > 0:
                    print(
                        f"New best model saved: reward {ep_reward:.2f} "
                        f"(previous {old_best:.2f})"
                    )
                    print(f"Main checkpoint: {self.model_path.with_suffix('.zip')}")
                    print(f"Archive checkpoint: {archive_model_path.with_suffix('.zip')}")

        return True
