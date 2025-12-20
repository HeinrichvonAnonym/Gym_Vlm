"""
Visualization script for Cartpole environment.
This script runs the environment with visualization enabled.
"""

import argparse
from utils.parse_task import parse_task, load_task_config
from utils.controllers import Controller
import torch

from agent.vlm.qwen_interface import LMP

def main():
    parser = argparse.ArgumentParser(description='Visualize Cartpole Environment')
    parser.add_argument('--num_envs', type=int, default=16, 
                        help='Number of environments to visualize (default: 16)')
    parser.add_argument('--policy', type=str, default='random', 
                        choices=['random', 'pd'],
                        help='Policy to use: random or pd (proportional-derivative)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps to run (default: 1000)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TASK VISUALIZATION")
    print("="*60)
    print(f"Number of environments: {args.num_envs}")
    print(f"Policy: {args.policy}")
    print(f"Max steps: {args.max_steps}")
    print()
    print("Controls:")
    print("  ESC - Exit")
    print("  V   - Toggle viewer sync")
    print("  R   - Start/stop recording frames")
    print("="*60)
    
    # Create task arguments
    class TaskArgs:
        def __init__(self):
            self.sim_device = 'cpu'
            self.rl_device = 'cpu'
            self.graphics_device_id = 0  # Use GPU 0 for rendering
            self.headless = False  # Enable visualization
            self.task = 'DualFranka'
    
    task_args = TaskArgs()
    
    # Load config and override num_envs
    cfg = load_task_config(task_args.task)
    cfg['env']['numEnvs'] = args.num_envs
    
    print("Initializing environment...")
    task = parse_task(task_args, cfg, cfg_train=None, sim_params=None)
    
    print(f"\nEnvironment initialized successfully!")
    print(f"Observation space: {task.num_obs}")
    print(f"Action space: {task.num_acts}")
    print(f"Device: {task.device}")
    print("\nStarting simulation...\n")

    vlm_cfg = cfg['agent']['vlm']

    lmp_interface = LMP(
        name=vlm_cfg['name'],
        cfg=vlm_cfg,
        fixed_vars={},
        variable_vars={},
        debug=False
    )

    
    # Reset environment
    obs = task.reset()
    
    # Define policy
    if args.policy == 'random':
        def policy(observations):
            """Random policy."""
            actions = torch.zeros((task.num_envs, task.num_acts), device=task.rl_device)
            actions[:, 1] = 0.1
            return actions
    else:  # pd policy
        def policy(observations):
            """Simple PD controller."""
            pole_angle = observations[:, 2]
            pole_vel = observations[:, 3]
            cart_pos = observations[:, 0]
            
            # PD control
            action = -(pole_angle * 10.0 + pole_vel * 1.0 + cart_pos * 0.5)
            action = torch.clamp(action, -1.0, 1.0).unsqueeze(-1)
            return action
    
    # Run simulation
    step = 0
    episode_rewards = []
    current_episode_reward = 0
    
    try:
        while step < args.max_steps:
            # Get action
            actions = policy(obs['obs'])
            
            # Step environment
            obs, rewards, dones, info = task.step(actions)

            task.process_vision_buffer()
            
            # Render
            task.render()
            
            # Track rewards
            current_episode_reward += rewards.mean().item()
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}/{args.max_steps} | "
                      f"Mean Reward: {rewards.mean().item():.3f} | "
                      f"Resets: {dones.sum().item()}")
            
            # Track episodes
            if dones.any():
                episode_rewards.append(current_episode_reward / 100)
                current_episode_reward = 0
            
            step += 1
    
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
    
    # Print statistics
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print(f"Total steps: {step}")
    if episode_rewards:
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average episode reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print("="*60)


if __name__ == "__main__":
    main()