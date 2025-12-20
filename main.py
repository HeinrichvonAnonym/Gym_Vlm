"""
Visualization script for Cartpole environment.
This script runs the environment with visualization enabled.
"""

import argparse
from utils.parse_task import  load_task_config
import torch
# from isaacgym_interface import IsaacGymVlmInterface
from isaacgym_env import IsaacGymEnv
from agent.vlm.qwen_interface import LMP
from utils.controllers import Controller
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Visualize Cartpole Environment')
    parser.add_argument('--num_envs', type=int, default=4, 
                        help='Number of environments to visualize (default: 1)')
    parser.add_argument('--policy', type=str, default='random', 
                        choices=['random', 'pd'],
                        help='Policy to use: random or pd (proportional-derivative)')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum steps to run (default: 10000)')
    
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
            self.sim_device = 'cuda:0'
            self.rl_device = 'cuda:0'
            self.graphics_device_id = 0  # Use GPU 0 for rendering
            self.headless = False  # Enable visualization
            self.task = 'MultiFranka'
    
    task_args = TaskArgs()
    
    # Load config and override num_envs
    cfg = load_task_config(task_args.task)

    
    env = IsaacGymEnv(cfg, args, task_args)
    print(env.get_object_names())

    # vlm_cfg = cfg['agent']['vlm']

    # lmp_interface = LMP(
    #     name=vlm_cfg['name'],
    #     cfg=vlm_cfg,
    #     fixed_vars={},
    #     variable_vars={},
    #     debug=False
    # )

    controller_cfg = cfg['agent']['controller']
    controller = Controller(env, controller_cfg)
    
    # # Reset environment
    descriptions, obs = env.reset()
    # print(obs.cam_2_rgb)

    # response = lmp_interface("grasp the cube")
    # print(f">>>>>>>>>>{response}>>>>>>>>>>>>>")
    # Define policy
    if args.policy == 'random':
        def policy(observations):
            """Random policy."""
            action = torch.zeros([env.task.num_envs, 7])
            action[:, 2] =0.0

            return action

    
    # Run simulation
    step = 0
    episode_rewards = []
    current_episode_reward = 0

    actions = np.array([ 0.1659,  0.1826,  1.1472, -0.8794,  0.4758, -0.0084,  0.0131, -1, -0.2659, -0.1826,  1.1472,  0.4758,  0.8794, -0.0131, -0.0084, -1])
    rel_actions = torch.zeros([1, 14])
    rel_actions[:, 0] = 0.1

    rel_actions[:, 7] = 0.1
    rel_actions[:, 9] = -0.1
    env.apply_action(actions)

    test_ori = -0.3
    
    try:
        while step < args.max_steps:
            # Get action

            
            # actions = policy(obs)
            
            # Step environment
            # print(actions.shape)
            # obs, rewards, dones, info = env.step(actions)
            # actionns = np.array([0.22633785,
            #                     -0.02424368,
            #                     1.17732811,  
            #                     0.99812859,
            #                     -0.00531844, 
            #                     -0.00883576,
            #                     0.06027498,
            #                     0.0])

            # env.apply_action(actionns)

            ee_poses = env.get_ee_pose()
            # print(ee_poses[0])
            # env.task.step(rel_actions)


            # print(ee_poses)
            if step % 100 == 0:
                env.task.process_vision_buffer()
            
            if step % 500 == 0:
                actions[2] -= test_ori
                test_ori *= -1
                env.apply_action(actions)
                

            # env.task.process_vision_buffer()

            # senv.task.process_vision_buffer()
            if ee_poses is not None:
                # print(ee_poses)
                pass
            
            # Render
            env.render()

            # env.get_scene_3d_obs(ignore_robot=False)
    
            
       
            
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