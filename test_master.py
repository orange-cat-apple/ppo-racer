import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def main():
    model_path = "./models_clean/best_model/best_model"
    video_dir = "./video_master"
    
    if not os.path.exists(model_path + ".zip"):
        print("Model not found.")
        return

    def make_env():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        return RecordVideo(
            env, 
            video_folder=video_dir,
            name_prefix="deepmind_portfolio_lap",
            episode_trigger=lambda x: True
        )

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_path, env=env)
    obs = env.reset()

    try:
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, _, _ = env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Video generated successfully.")

if __name__ == "__main__":
    main()