import os
import time
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

import hivemind_env  # Registers HiveMind-SingleAgent environment
from hivemind_env.models import CustomCombinedExtractor


def dynamic_linear_lr_schedule(initial_lr: float = 3e-4, min_lr: float = 3e-5):
    """
    Linear learning rate decay schedule (3e-4 down to 3e-5).
    """
    def func(progress_remaining: float) -> float:
        return min_lr + progress_remaining * (initial_lr - min_lr)
    return func


class TrainingCheckpointCallback(BaseCallback):
    """
    Saves periodic checkpoints to disk during training.
    """
    def __init__(self, check_freq: int = 5000, save_path: str = "./models", verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            latest_model_path = os.path.join(self.save_path, "ppo_hivemind_latest.zip")
            self.model.save(latest_model_path)
            lr = self.model.lr_schedule(self.model._current_progress_remaining)
            print(f"[Step {self.num_timesteps}] Saved checkpoint to {latest_model_path} | LR: {lr:.6f}")
        return True


def make_env(rank: int, seed: int = None):
    def _init():
        import hivemind_env  # Required inside worker process for SubprocVecEnv
        env = gym.make("HiveMind-SingleAgent", render_mode="DIRECT")
        env_seed = (seed + rank) if seed is not None else None
        env.reset(seed=env_seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="HiveMind Pure PPO Training Pipeline")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible training (default: Truly Random per run)")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps to train (default: 500000)")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments for workstation CPU multi-processing (default: 8)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: 'cuda', 'cpu', or 'auto' (default: auto)")
    args = parser.parse_args()

    print("=========================================================================")
    print("      HiveMind Pure PPO Training Pipeline (Original Solid Baseline)      ")
    print("=========================================================================")

    TOTAL_TIMESTEPS = args.timesteps
    NUM_ENVS = args.num_envs
    SEED = args.seed

    # 1. Device Selection (NVIDIA GPU / CUDA Acceleration)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[*] [GPU ACCELERATION ACTIVATED] Device: {device.upper()} | Model: {gpu_name}")
        torch.backends.cudnn.benchmark = True
    else:
        print(f"[*] [COMPUTE DEVICE] Running on: CPU")
        if not torch.cuda.is_available():
            print("    Notice: PyTorch CUDA is not detected in this environment.")
            print("    To enable NVIDIA GPU acceleration, install CUDA-enabled PyTorch:")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu121\n")

    if SEED is not None:
        print(f"[*] Training with Fixed Reproducible Base Seed: {SEED}")
    else:
        print("[*] Training with Dynamic System Entropy (Truly Random World Generation Per Episode)")

    print(f"[*] Parallel CPU Worker Processes: {NUM_ENVS}")

    MODEL_DIR = "./models"
    LOG_DIR = "./tensorboard_logs"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Vectorized environments: SubprocVecEnv for parallel worker processes
    env_fns = [make_env(rank=i, seed=SEED) for i in range(NUM_ENVS)]
    if NUM_ENVS > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64])
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=dynamic_linear_lr_schedule(initial_lr=3e-4, min_lr=3e-5),
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=LOG_DIR,
        verbose=1,
        device=device,
        seed=SEED
    )

    callback = TrainingCheckpointCallback(check_freq=5000, save_path=MODEL_DIR)

    print(f"\nStarting PPO training for {TOTAL_TIMESTEPS} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60.0:.2f} minutes.")

    final_model_path = os.path.join(MODEL_DIR, "ppo_hivemind_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    vec_env.close()

if __name__ == "__main__":
    main()
