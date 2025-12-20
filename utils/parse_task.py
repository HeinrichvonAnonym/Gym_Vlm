"""
Parse task configuration and create environment instances.
"""
import os
import yaml
from typing import Dict, Any
from env.task.vec_task import VecTask
from env.task.cartpole import Cartpole
from env.task.multi_franka import MultiFranka


def parse_task(args, cfg, cfg_train, sim_params):
    """
    Parse task configuration and return the appropriate task class and configuration.
    
    Args:
        args: Command line arguments
        cfg: Task configuration dictionary
        cfg_train: Training configuration dictionary
        sim_params: Simulation parameters
        
    Returns:
        task: Instantiated task object
    """

    
    # Task registry - add new tasks here
    task_map = {
        "Cartpole": Cartpole,
        "MultiFranka": MultiFranka,
        # Add more tasks as needed
        # "AnotherTask": AnotherTaskClass,
    }
    
    # Get task name from config
    task_name = cfg["name"]
    
    # Check if task exists in registry
    if task_name not in task_map:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(task_map.keys())}")
    
    # Get the task class
    task_class = task_map[task_name]
    
    # Extract environment configuration
    env_config = {
        "env": cfg["env"],
        "sim": cfg["sim"],
        "physics_engine": cfg["physics_engine"],
    }
    
    # Set device configuration
    sim_device = args.sim_device if hasattr(args, 'sim_device') else 'cpu'
    rl_device = args.rl_device if hasattr(args, 'rl_device') else 'cpu'
    graphics_device_id = args.graphics_device_id if hasattr(args, 'graphics_device_id') else -1
    headless = args.headless if hasattr(args, 'headless') else True
    
    # Instantiate the task
    task = task_class(
        cfg=env_config,
        rl_device=rl_device,
        sim_device=sim_device,
        graphics_device_id=graphics_device_id,
        headless=headless
    )
    
    return task


def load_task_config(task_name: str, config_dir: str = "cfg") -> Dict[str, Any]:
    """
    Load task configuration from YAML file.
    
    Args:
        task_name: Name of the task
        config_dir: Directory containing config files
        
    Returns:
        Configuration dictionary
    """
    config_path = os.path.join(config_dir, "task", f"{task_name}.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def create_task_from_config(task_name: str, sim_device: str = 'cpu', 
                            rl_device: str = 'cpu', headless: bool = True):
    """
    Convenience function to create a task from its name.
    
    Args:
        task_name: Name of the task
        sim_device: Device for simulation ('cpu' or 'cuda:0')
        rl_device: Device for RL algorithm
        headless: Whether to run without rendering
        
    Returns:
        Instantiated task object
    """
    # Load configuration
    cfg = load_task_config(task_name)
    
    # Create simple args object
    class Args:
        pass
    
    args = Args()
    args.sim_device = sim_device
    args.rl_device = rl_device
    args.graphics_device_id = -1 if headless else 0
    args.headless = headless
    
    # Create and return task
    return parse_task(args, cfg, None, None)