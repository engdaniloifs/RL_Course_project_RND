from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
import ale_py  # registers Atari envs
from gymnasium.wrappers import TimeLimit,TransformAction
from gymnasium.spaces import Discrete
import gymnasium as gym

def make_custom_env(env_id: str, seed: int = 0):
    

    def _init():
        env = gym.make(
            env_id,
            full_action_space=False,
            render_mode="human",
        )
        env.reset(seed=seed)
        allowed_actions = [0, 1, 2, 3, 4, 5, 11, 12]
        env = TransformAction(
            env,
            func=lambda a: allowed_actions[a],
            action_space=Discrete(len(allowed_actions)),
        )
        env = AtariWrapper(
            env,
            terminal_on_life_loss=False,
            clip_reward=False,
        )
        return env

    return _init



def watch_trained(model_path: str, env_id: str, seed: int = 0):
    venv = DummyVecEnv([make_custom_env(env_id, seed)])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)
    #print observation space and action space
    print(f"Observation space: {venv.observation_space}")

    model = PPO.load(model_path)

    obs = venv.reset()
    print("Watching trained model...")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = venv.step(action)

            if dones[0]:
                obs = venv.reset()
    finally:
        venv.close()


def main():
    watch_trained(
        "checkpoints/ppo_rnd_montezuma_4000000_steps.zip",
        "MontezumaRevengeNoFrameskip-v4",
        seed=0,
    )

if __name__ == "__main__":
    main()