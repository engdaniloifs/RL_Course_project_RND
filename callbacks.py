import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rnd import RNDModel
from envs import make_video_env
from pathlib import Path
import json
from datetime import datetime
import pickle


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

            self.logger.record("test/max_reward", float(self.best_reward))
        
        current_reward = max(ep_info["r"] for ep_info in self.model.ep_info_buffer) if len(self.model.ep_info_buffer) > 0 else 0.0
        self.logger.record("episode/r_ext", float(current_reward))

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
        self.best_episode_max_room = float("-inf")
        self.best_live_room = float("-inf")
        self.best_episode_reward = float("-inf")
        

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

                if room > self.best_live_room:
                    self.best_live_room = room

        self.logger.record("rooms/total_unique_rooms", len(self.global_rooms))

        if self.best_live_room != float("-inf"):
            self.logger.record("rooms/best_live_room", self.best_live_room)

        for i, done in enumerate(dones):
            if done:
                visited = self.local_rooms[i]

                if visited:
                    episode_max_room = max(visited)
                    self.logger.record("rooms/episode_max_room", episode_max_room)

                    if episode_max_room > self.best_episode_max_room:
                        self.best_episode_max_room = episode_max_room

                ep_info = infos[i].get("episode")
                if ep_info is not None:
                    r_ext = float(ep_info["r"])
                    ep_len = int(ep_info["l"])

                    self.logger.record("episode/r_ext", r_ext)
                    self.logger.record("episode/length", ep_len)
                    self.logger.record("episode/success", float(r_ext > 0.0))

                    if r_ext > self.best_episode_reward:
                        self.best_episode_reward = r_ext

                self.local_rooms[i].clear()

        if self.best_episode_max_room != float("-inf"):
            self.logger.record("rooms/best_episode_max_room", self.best_episode_max_room)

        if self.best_episode_reward != float("-inf"):
            self.logger.record("episode/best_r_ext", self.best_episode_reward)

        return True



class BestPolicySaverCallback(BaseCallback):
    def __init__(self, save_path="checkpoints", verbose=0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.archive_path = self.save_path / "archive"
        self.archive_path.mkdir(parents=True, exist_ok=True)

        self.best_path = self.save_path / "best"
        self.best_path.mkdir(parents=True, exist_ok=True)

        self.room_progress_path = self.save_path / "room_progress"
        self.room_progress_path.mkdir(parents=True, exist_ok=True)

        self.best_meta_path = self.best_path / "best_episode.json"
        self.best_model_path = self.best_path / "best_model"

        self.best_episode_reward = float("-inf")

        self.local_rooms = None
        self.should_record_room_episode = None
        self.record_trigger_room_count = None

        self._load_previous_best()

    def _load_previous_best(self) -> None:
        if not self.best_meta_path.exists():
            return

        try:
            with open(self.best_meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.best_episode_reward = float(metadata.get("reward", float("-inf")))
            if self.verbose > 0:
                print(f"Loaded previous best reward: {self.best_episode_reward:.2f}")
        except Exception as e:
            if self.verbose > 0:
                print(f"Could not load previous best metadata: {e}")

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.local_rooms = [set() for _ in range(n_envs)]
        self.should_record_room_episode = [False for _ in range(n_envs)]
        self.record_trigger_room_count = [0 for _ in range(n_envs)]

    def _save_pickle(self, path: Path, payload: dict) -> None:
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _save_json(self, path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _build_episode_blob(self, info: dict, visited_rooms) -> dict:
        return {
            "actions": info.get("episode_actions"),
            "rewards": info.get("episode_rewards"),
            "episode_start_state": info.get("episode_start_state"),
            "visited_rooms": sorted(list(visited_rooms)),
            "num_rooms_visited": len(visited_rooms),
            "final_room": info.get("room"),
            "episode_seed": info.get("episode_seed"),
            "episode_idx": info.get("episode_idx"),
            "env_rank": info.get("env_rank"),
        }

    def _save_best_episode(self, ep_reward, ep_info, env_idx, info, visited_rooms) -> None:
        old_best = self.best_episode_reward
        self.best_episode_reward = ep_reward

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reward_str = f"{ep_reward:.2f}".replace(".", "p")

        # overwrite latest best model
        self.model.save(str(self.best_model_path))

        archive_model_path = (
            self.archive_path
            / f"best_model_reward_{reward_str}_step_{self.num_timesteps}_{timestamp}"
        )
        self.model.save(str(archive_model_path))

        episode_blob = self._build_episode_blob(info, visited_rooms)

        payload_path = (
            self.archive_path
            / f"best_episode_payload_reward_{reward_str}_step_{self.num_timesteps}_{timestamp}.pkl"
        )
        self._save_pickle(payload_path, episode_blob)

        metadata = {
            "type": "best_episode",
            "reward": float(ep_reward),
            "length": int(ep_info["l"]),
            "num_timesteps": int(self.num_timesteps),
            "env_idx": int(env_idx),
            "episode_seed": info.get("episode_seed"),
            "episode_idx": info.get("episode_idx"),
            "env_rank": info.get("env_rank"),
            "visited_rooms": sorted(list(visited_rooms)),
            "num_rooms_visited": len(visited_rooms),
            "saved_at": timestamp,
            "main_model_path": str(self.best_model_path.with_suffix(".zip")),
            "archive_model_path": str(archive_model_path.with_suffix(".zip")),
            "payload_path": str(payload_path),
        }

        self._save_json(self.best_meta_path, metadata)

        archive_meta_path = (
            self.archive_path
            / f"best_episode_reward_{reward_str}_step_{self.num_timesteps}_{timestamp}.json"
        )
        self._save_json(archive_meta_path, metadata)

        if self.verbose > 0:
            print(
                f"New best model saved: reward {ep_reward:.2f} "
                f"(previous {old_best:.2f})"
            )
            print(f"Best payload saved at: {payload_path}")

    def _save_room_progress_episode(self, ep_info, env_idx, info, visited_rooms) -> None:
        """
        Archive any completed episode that visited more than one room.
        This is NOT tied to best reward.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_rooms = len(visited_rooms)
        max_room = max(visited_rooms) if visited_rooms else None

        archive_model_path = (
            self.room_progress_path
            / f"room_progress_model_rooms_{num_rooms}_maxroom_{max_room}_step_{self.num_timesteps}_{timestamp}"
        )
        self.model.save(str(archive_model_path))

        episode_blob = self._build_episode_blob(info, visited_rooms)

        payload_path = (
            self.room_progress_path
            / f"room_progress_payload_rooms_{num_rooms}_maxroom_{max_room}_step_{self.num_timesteps}_{timestamp}.pkl"
        )
        self._save_pickle(payload_path, episode_blob)

        metadata = {
            "type": "room_progress_episode",
            "reward": float(ep_info["r"]),
            "length": int(ep_info["l"]),
            "num_timesteps": int(self.num_timesteps),
            "env_idx": int(env_idx),
            "episode_seed": info.get("episode_seed"),
            "episode_idx": info.get("episode_idx"),
            "env_rank": info.get("env_rank"),
            "visited_rooms": sorted(list(visited_rooms)),
            "num_rooms_visited": num_rooms,
            "max_room": max_room,
            "saved_at": timestamp,
            "archive_model_path": str(archive_model_path.with_suffix(".zip")),
            "payload_path": str(payload_path),
        }

        archive_meta_path = (
            self.room_progress_path
            / f"room_progress_episode_rooms_{num_rooms}_maxroom_{max_room}_step_{self.num_timesteps}_{timestamp}.json"
        )
        self._save_json(archive_meta_path, metadata)

        if self.verbose > 0:
            print(
                f"Room-progress episode saved: env {env_idx}, "
                f"{num_rooms} rooms visited, max room {max_room}"
            )
            print(f"Room-progress payload saved at: {payload_path}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for env_idx, info in enumerate(infos):
            room = info.get("room")
            if room is not None:
                self.local_rooms[env_idx].add(int(room))

                # mark this episode for mandatory archival once it finishes
                if len(self.local_rooms[env_idx]) > 1:
                    self.should_record_room_episode[env_idx] = True
                    self.record_trigger_room_count[env_idx] = len(self.local_rooms[env_idx])

        for env_idx, info in enumerate(infos):
            if not dones[env_idx]:
                continue

            ep_info = info.get("episode")
            visited_rooms = set(self.local_rooms[env_idx])

            # Rule 1: best run so far
            if ep_info is not None:
                ep_reward = float(ep_info["r"])
                if ep_reward > self.best_episode_reward:
                    self._save_best_episode(ep_reward, ep_info, env_idx, info, visited_rooms)

            # Rule 2: any episode that reached the next room / visited >1 room
            if self.should_record_room_episode[env_idx]:
                if ep_info is not None:
                    self._save_room_progress_episode(ep_info, env_idx, info, visited_rooms)

            # reset env-local trackers after episode ends
            self.local_rooms[env_idx].clear()
            self.should_record_room_episode[env_idx] = False
            self.record_trigger_room_count[env_idx] = 0

        return True
