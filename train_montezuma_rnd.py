from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from callbacks import RNDBonusCallback,BestPolicySaverCallback, RoomLoggerCallback
from envs import make_env
from rnd import RNDModel
import ale_py
#tensorboard --logdir=./logs

ENV_ID = "MontezumaRevengeNoFrameskip-v4"
N_ENVS = 16
SEED = 0
DEVICE = "cuda"

TOTAL_TIMESTEPS = 10_000_000
N_STEPS = 256
BATCH_SIZE = 512
N_EPOCHS = 4
GAMMA = 0.999
GAE_LAMBDA = 0.95

CLIP_RANGE = 0.1
ENT_COEF = 0.001

LEARNING_RATE = 5e-5
RND_LR = 5e-4
RND_LATENT_DIM = 512
RND_UPDATE_PROPORTION = 1.0
intrinsic_coefficient = 0.50
extrinsic_coefficient = 1.0

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

    rnd_callback = RNDBonusCallback(rnd_model=rnd, intrinsic_coefficient=intrinsic_coefficient, extrinsic_coefficient=extrinsic_coefficient, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path="./checkpoints_skulltrue/",
        name_prefix="ppo_rnd_montezuma",
    )
    best_callback = BestPolicySaverCallback(
        save_path="./best_models_skulltrue/",
        verbose=1,
    )

    callbacks = CallbackList([
        rnd_callback,                       # modifies rewards (important: first)
        RoomLoggerCallback(verbose=1),      # reads info, logs
        best_callback,                      # saves best model (needs episode info)
        checkpoint_callback,                # periodic saves (last)
    ])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        tb_log_name="new_run",
    )
    model.save("ppo_rnd_montezuma_minimal_skull_true")


if __name__ == "__main__":
    main()