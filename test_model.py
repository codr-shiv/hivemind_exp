import os
import time
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
import pybullet as pb
import hivemind_env

def main():
    parser = argparse.ArgumentParser(description="HiveMind Agent Evaluation")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation (default: Truly Random per episode)")
    parser.add_argument("--direct", action="store_true", help="Run in DIRECT mode without GUI window")
    args = parser.parse_args()

    print("=========================================================")
    print("   HiveMind Trained Agent Evaluation & Visual Demo      ")
    print("=========================================================")

    # 1. Locate trained policy checkpoint
    model_path = "./models/ppo_hivemind_final.zip"
    if not os.path.exists(model_path):
        model_path = "./models/ppo_hivemind_latest.zip"

    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at {model_path}. Running evaluation with random policy.")
        model = None
    else:
        try:
            print(f"Loading trained PPO model from: {model_path}")
            model = PPO.load(model_path)
        except Exception as e:
            print(f"Notice: Could not load checkpoint from {model_path} ({e}). Evaluating with fresh policy.")
            model = None

    # 2. Initialize environment
    render_mode = "DIRECT" if args.direct else "human"
    env = gym.make("HiveMind-SingleAgent", render_mode=render_mode)
    unwrapped = env.unwrapped

    for episode in range(1, args.episodes + 1):
        ep_seed = (args.seed + episode - 1) if args.seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        
        # Extract map details for verification
        r_pos, _ = pb.getBasePositionAndOrientation(unwrapped.robot_id, physicsClientId=unwrapped.client_id)
        res_pos, _ = pb.getBasePositionAndOrientation(unwrapped.resource_id, physicsClientId=unwrapped.client_id)
        r_grid = unwrapped._world_to_grid(r_pos[0], r_pos[1])
        res_grid = unwrapped._world_to_grid(res_pos[0], res_pos[1])
        dep_grid = unwrapped.depot_pos_grid
        num_obs = len(unwrapped.obstacle_ids)

        done = False
        total_reward = 0.0
        step_count = 0
        print(f"\n--- Starting Evaluation Episode {episode}/{args.episodes} ---")
        print(f"    Layout: Robot Grid={r_grid} | Resource Grid={res_grid} | Depot Grid={dep_grid} | Obstacles={num_obs}")

        while not done:
            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            if render_mode == "human":
                time.sleep(0.03)

        status = "SUCCESS (Resource Delivered!)" if total_reward > 5.0 else "FINISHED"
        print(f"Episode {episode} {status} | Steps: {step_count} | Total Reward: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
