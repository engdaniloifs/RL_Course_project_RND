import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rnd import RNDModel
from envs import make_video_env


class RNDBonusCallback(BaseCallback):
    def __init__(self, rnd_model: RNDModel,intrisic_coefficient, extrinsic_coefficient, verbose=0):
        super().__init__(verbose)
        self.rnd = rnd_model
        self.rollout_next_obs = []
        self.intrisinc_coefficient = intrisic_coefficient
        self.extrinsic_coefficient = extrinsic_coefficient

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
        self.rnd.obs_rms.update(obs_float)

        tgt_full = self.rnd.compute_target_features(norm_obs)
        intrinsic = self.rnd.compute_intrinsic_reward(norm_obs, tgt=tgt_full)
        intrinsic = intrinsic.reshape(n_steps, n_envs)

        self.model.rollout_buffer.rewards = self.model.rollout_buffer.rewards * self.extrinsic_coefficient + intrinsic * self.intrinsic_coefficient

        last_values = self.locals["values"]
        dones = self.locals["dones"]
        self.model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=dones,
        )

        rnd_loss = self.rnd.update(norm_obs, tgt_full=tgt_full)

        self.logger.record("rnd/loss", float(rnd_loss))
        self.logger.record("rnd/intrinsic_mean", float(intrinsic.mean()))
        self.logger.record("rnd/intrinsic_std", float(intrinsic.std()))
        self.logger.record("rnd/intrinsic_max", float(intrinsic.max()))

from stable_baselines3.common.callbacks import BaseCallback


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

                print(f"Env {i} visited rooms: {sorted(visited)}")
                self.local_rooms[i].clear()

        return True


class VideoRecorderCallback(BaseCallback):
    def __init__(self, save_freq=100_000, video_length=4000, deterministic=False, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.video_length = video_length
        self.deterministic = deterministic
        self.last_recorded_at = 0
        self.best_episode_reward = 0.5

    def _on_step(self) -> bool:
        target = (self.num_timesteps // self.save_freq) * self.save_freq
        if target > self.last_recorded_at:
            self.last_recorded_at = target
            self._record_video(tag=f"step_{target}")

        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            ep_reward = ep_info["r"]
            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                self._record_video(tag=f"best_{int(ep_reward)}_step_{self.num_timesteps}")

        return True

    def _record_video(self, tag: str) -> None:
        video_env = make_video_env(tag)
        obs = video_env.reset()
        steps = 0

        while steps < self.video_length:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, dones, _ = video_env.step(action)
            steps += 1

            if dones[0]:
                obs = video_env.reset()

        video_env.close()