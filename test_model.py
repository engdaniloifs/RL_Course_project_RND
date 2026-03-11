from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3 import PPO
import ale_py  # registers Atari envs


def watch_trained(model_path: str, env_id: str, seed: int = 0):
    venv = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": "human"},
        wrapper_kwargs={"terminal_on_life_loss": False},
    )
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)

    model = PPO.load(model_path)

    obs = venv.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)

            if dones[0]:
                obs = venv.reset()
    finally:
        venv.close()


def main():
    watch_trained(
        "checkpoints/ppo_rnd_montezuma_1000000_steps.zip",
        "MontezumaRevengeNoFrameskip-v4",
        seed=0,
    )

if __name__ == "__main__":
    main()