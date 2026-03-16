from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from callbacks import RNDBonusCallback, VideoRecorderCallback
from envs import make_env
from rnd import RNDModel
import ale_py


ENV_ID = "MontezumaRevengeNoFrameskip-v4"
N_ENVS = 32
SEED = 0
DEVICE = "cuda"

TOTAL_TIMESTEPS = 10_000_000
N_STEPS = 128
BATCH_SIZE = 1024
N_EPOCHS = 4
GAMMA = 0.999
GAE_LAMBDA = 0.95
LEARNING_RATE = 1e-4
CLIP_RANGE = 0.1
ENT_COEF = 0.001

RND_LR = 1e-4
RND_LATENT_DIM = 512
RND_UPDATE_PROPORTION = 1.0
INT_REWARD_COEF = 1.0

CHECKPOINT_FREQ = 1_000_000 // N_ENVS
VIDEO_FREQ = 1_000_000
VIDEO_LENGTH = 4000


def main():
    env = make_env(n_envs=N_ENVS, seed=SEED)

    rnd = RNDModel(
        obs_shape=env.observation_space.shape,
        device=DEVICE,
        lr=RND_LR,
        latent_dim=RND_LATENT_DIM,
        update_proportion=RND_UPDATE_PROPORTION,
        int_reward_coef=INT_REWARD_COEF,
    )

    model = PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        verbose=1,
        tensorboard_log="./tb_logs/",
        device=DEVICE,
    )

    rnd_callback = RNDBonusCallback(rnd_model=rnd, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./checkpoints/",
        name_prefix="ppo_rnd_montezuma",
    )

    video_callback = VideoRecorderCallback(
        save_freq=VIDEO_FREQ,
        video_length=VIDEO_LENGTH,
        deterministic=False,
        verbose=1,
    )

    callbacks = CallbackList([
        rnd_callback,
        checkpoint_callback,
        video_callback,
    ])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="ppo_rnd_montezuma",
    )
    model.save("ppo_rnd_montezuma_minimal")


if __name__ == "__main__":
    main()