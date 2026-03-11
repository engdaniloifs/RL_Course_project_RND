import gymnasium as gym
import ale_py
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage,VecVideoRecorder, VecMonitor,DummyVecEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.atari_wrappers import AtariWrapper

# -----------------------------
# Running mean/std for obs/reward normalization
# -----------------------------
class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# -----------------------------
# RND encoder
# Input shape after VecTransposeImage:
# (C, H, W), typically (4, 84, 84)
# -----------------------------
class RNDEncoder(nn.Module):
    def __init__(self, obs_shape, latent_dim=512):
        super().__init__()
        c, h, w = obs_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, c, h, w)).shape[1]

        self.head = nn.Linear(n_flatten, latent_dim)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.cnn(x)
        return self.head(x)


class RNDModel:
    def __init__(
        self,
        obs_shape,
        device="cuda",
        lr=1e-4,
        latent_dim=512,
        update_proportion=0.25,
        int_reward_coef=1.0,
    ):
        rnd_obs_shape = ( 1, obs_shape[1], obs_shape[2])  # only last frame channel
        self.device = th.device(device if th.cuda.is_available() else "cpu")
        self.predictor = RNDEncoder(rnd_obs_shape, latent_dim).to(self.device)
        self.target = RNDEncoder(rnd_obs_shape, latent_dim).to(self.device)

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.update_proportion = update_proportion
        self.int_reward_coef = int_reward_coef

        # Normalize only the last observations seen by RND
        self.obs_rms = RunningMeanStd(shape=rnd_obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))

    def _normalize_obs(self, obs_float: np.ndarray) -> np.ndarray:
        # obs: (N, C, H, W)
        # OpenAI-style RND often normalizes the last frame channel only.
        

        mean = self.obs_rms.mean
        std = np.sqrt(self.obs_rms.var + 1e-8)
        obs_norms = (obs_float - mean) / std
        return np.clip(obs_norms, -5.0, 5.0)
    def compute_target_features(self, norm_obs: np.ndarray) -> th.Tensor:
        obs_t = th.as_tensor(norm_obs, device=self.device, dtype=th.float32)
        with th.no_grad():
            tgt = self.target(obs_t)
        return tgt

    def compute_intrinsic_reward(self, norm_obs: np.ndarray, tgt: th.Tensor | None = None) -> np.ndarray:
        obs_t = th.as_tensor(norm_obs, device=self.device, dtype=th.float32)

        with th.no_grad():
            pred = self.predictor(obs_t)
            if tgt is None:
                tgt = self.target(obs_t)

            reward = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)

        reward_raw = reward.detach().cpu().numpy()

        self.int_reward_rms.update(reward_raw)
        reward_np = reward_raw / np.sqrt(self.int_reward_rms.var + 1e-8)
         
        return self.int_reward_coef * reward_np

    def update(self, norm_obs: np.ndarray, tgt_full: th.Tensor | None = None, batch_size=256) -> float:
        dataset = th.as_tensor(norm_obs, device=self.device, dtype=th.float32)
        n = dataset.shape[0]
        perm = th.randperm(n, device=self.device)

        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs = dataset[idx]

            pred = self.predictor(obs)

            if tgt_full is None:
                with th.no_grad():
                    tgt = self.target(obs)
            else:
                tgt = tgt_full[idx]

            loss_per_sample = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)

            mask = (th.rand_like(loss_per_sample) < self.update_proportion).float()
            loss = (loss_per_sample * mask).sum() / th.clamp(mask.sum(), min=1.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0


class RNDBonusCallback(BaseCallback):
    """
    Minimal on-policy integration:
    - stores next observations during rollout
    - at rollout end, computes intrinsic rewards
    - adds intrinsic rewards directly to rollout_buffer.rewards
    - updates RND predictor
    """

    def __init__(self, rnd_model: RNDModel, verbose=0):
        super().__init__(verbose)
        self.rnd = rnd_model
        self.rollout_next_obs = []

    def _on_rollout_start(self) -> None:
        self.rollout_next_obs = []

    def _on_step(self) -> bool:
        # new_obs shape: (n_envs, C, H, W)
        new_obs = self.locals["new_obs"]          # shape: (n_envs, C, H, W)
        dones = self.locals["dones"]              # shape: (n_envs,)
        infos = self.locals["infos"]              # list of dicts

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
        

        norm_obs = self.rnd._normalize_obs(obs_float)
        self.rnd.obs_rms.update(obs_float)
        

        # cache target once
        tgt_full = self.rnd.compute_target_features(norm_obs)

        intrinsic = self.rnd.compute_intrinsic_reward(norm_obs, tgt=tgt_full)
        intrinsic = intrinsic.reshape(n_steps, n_envs)

        self.model.rollout_buffer.rewards += intrinsic
        last_values = self.locals["values"]
        dones = self.locals["dones"]
        
        self.model.rollout_buffer.compute_returns_and_advantage(
                last_values=last_values,
                dones=dones,
            )

        rnd_loss = self.rnd.update(norm_obs, tgt_full=tgt_full)
        

        if self.verbose > 0:
            print(f"RND loss: {rnd_loss:.5f}, intrinsic mean: {intrinsic.mean():.5f}")
            self.logger.record("rnd/loss", float(rnd_loss))
            self.logger.record("rnd/intrinsic_mean", float(intrinsic.mean()))
            self.logger.record("rnd/intrinsic_std", float(intrinsic.std()))
            self.logger.record("rnd/intrinsic_max", float(intrinsic.max()))

class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        save_freq=100_000,
        video_length=4000,
        deterministic=False,
        reward_threshold=100,
        verbose=0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.video_length = video_length
        self.deterministic = deterministic
        self.reward_threshold = reward_threshold
        self.last_recorded_at = 0
        self.best_episode_reward = 0.5
        self.recorded_first_threshold = False

    def _on_step(self) -> bool:
        # 1) periodic video every 100k steps
        target = (self.num_timesteps // self.save_freq) * self.save_freq
        if target > self.last_recorded_at:
            self.last_recorded_at = target
            self._record_video(tag=f"step_{target}")

        # 2) event-based video when a finished episode has better reward
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            ep_reward = ep_info["r"]

            # Record when a new best episode appears
            if ep_reward > self.best_episode_reward:
                self.best_episode_reward = ep_reward
                self._record_video(tag=f"best_{int(ep_reward)}_step_{self.num_timesteps}")

        return True

    def _record_video(self, tag: str):
        video_env = make_video_env(tag)
        obs = video_env.reset()
        steps = 0
        while steps < self.video_length:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, _, done, _ = video_env.step(action)
            steps += 1

            if done[0]:
                obs = video_env.reset()
        

        video_env.close()

        if self.verbose > 0:
            print(f"Recorded video: {tag}")





def make_single_env(seed: int, rank: int):
    def _init():
        env = gym.make("MontezumaRevengeNoFrameskip-v4")

        # Seed each env differently
        env.reset(seed=seed + rank)

        # Atari preprocessing
        env = AtariWrapper(
            env,
            terminal_on_life_loss=False,  # your config
        )

        # 18k frames / skip 4 = 4500 env steps
        env = TimeLimit(env, max_episode_steps=4500)

        return env

    return _init


def make_env(n_envs: int = 16, seed: int = 0):
    env = DummyVecEnv([make_single_env(seed, i) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env)
    return env


def make_video_env(step_tag):
    env = make_atari_env(
        "MontezumaRevengeNoFrameskip-v4",
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

if __name__ == "__main__":
    env = make_env()    

    obs_shape = env.observation_space.shape  # expected (4, 84, 84)
    rnd = RNDModel(
        obs_shape=obs_shape,
        device="cuda",
        lr=1e-4,
        latent_dim=512,
        update_proportion=0.25,
        int_reward_coef=1.0,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,          # often higher for sparse Atari
        gae_lambda=0.95,
        learning_rate=1e-4,
        clip_range=0.1,
        ent_coef=0.001,
        verbose=1,
        tensorboard_log="./tb_logs/",
        device="cuda",
        
    )

    callback = RNDBonusCallback(rnd_model=rnd, verbose=1)
    checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000 // 16,
    save_path="./checkpoints/",
    name_prefix="ppo_rnd_montezuma"
)
    video_callback = VideoRecorderCallback(
    save_freq=100_000,
    video_length=4000,
    deterministic=False,
    verbose=1,
)
    callbacks = CallbackList([
        callback,
        checkpoint_callback,
        video_callback,
    ])
    model.learn(total_timesteps=20_000_000, callback=callbacks,tb_log_name="ppo_rnd_montezuma")
    model.save("ppo_rnd_montezuma_minimal")