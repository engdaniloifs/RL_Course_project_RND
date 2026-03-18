import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


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
        return self.head(self.cnn(x))


class RNDModel:
    def __init__(
        self,
        obs_shape,
        device="cuda",
        lr=1e-4,
        latent_dim=512,
        update_proportion=0.25
    ):
        rnd_obs_shape = (1, obs_shape[1], obs_shape[2])
        self.device = th.device(device if th.cuda.is_available() else "cpu")

        self.predictor = RNDEncoder(rnd_obs_shape, latent_dim).to(self.device)
        self.target = RNDEncoder(rnd_obs_shape, latent_dim).to(self.device)
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.update_proportion = update_proportion

        self.obs_rms = RunningMeanStd(shape=rnd_obs_shape)
        self.int_reward_rms = RunningMeanStd(shape=(1,))

    def normalize_obs(self, obs_float: np.ndarray) -> np.ndarray:
        mean = self.obs_rms.mean
        std = np.sqrt(self.obs_rms.var + 1e-8)
        obs_norm = obs_float - mean
        obs_norm = obs_norm/ std
        return np.clip(obs_norm, -5.0, 5.0)

    def compute_target_features(self, norm_obs: np.ndarray) -> th.Tensor:
        obs_t = th.as_tensor(norm_obs, device=self.device, dtype=th.float32)
        with th.no_grad():
            return self.target(obs_t)

    def compute_intrinsic_reward(
        self,
        norm_obs: np.ndarray,
        tgt: th.Tensor | None = None,
    ) -> np.ndarray:
        obs_t = th.as_tensor(norm_obs, device=self.device, dtype=th.float32)

        with th.no_grad():
            pred = self.predictor(obs_t)
            if tgt is None:
                tgt = self.target(obs_t)
            reward = F.mse_loss(pred, tgt, reduction="none").mean(dim=1)

        reward_raw = reward.detach().cpu().numpy()
        self.int_reward_rms.update(reward_raw)
        reward_norm = reward_raw / np.sqrt(self.int_reward_rms.var + 1e-8)
        return reward_norm

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