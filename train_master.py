import os
from typing import Callable
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env():
    return gym.make("CarRacing-v3", render_mode="rgb_array")

def main():
    model_dir = "./models_clean/checkpoints"
    log_dir = "./logs_clean"
    best_model_dir = "./models_clean/best_model"
    final_model_path = "./models_clean/ppo_carracing_final"
    
    for path in [model_dir, log_dir, best_model_dir]:
        os.makedirs(path, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=4)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=linear_schedule(1e-4),
        n_steps=512,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.15),
        ent_coef=0.005,
        verbose=1,
        tensorboard_log=log_dir
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=model_dir,
        name_prefix="ppo_carracing"
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=12500,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )

    print("Starting clean slate training. Target: 1,000,000 steps.")
    
    model.learn(
        total_timesteps=1000000,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        tb_log_name="ppo_master_run"
    )

    model.save(final_model_path)
    vec_env.close()
    eval_env.close()
    print("Training complete.")

if __name__ == "__main__":
    main()