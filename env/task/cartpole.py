"""
Simple Cartpole task for Isaac Gym - CPU compatible (Simplified)
This version requires the URDF file to exist.
"""
import os
import numpy as np
import torch

# Import at module level - this is critical
from isaacgym import gymapi
from isaacgym import gymtorch

# Import base class
from env.task.vec_task import VecTask

import cv2


class Cartpole(VecTask):
    """
    Simple Cartpole balancing task.
    The goal is to balance a pole on a cart by applying forces to the cart.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        
        # Task-specific parameters
        self.max_episode_length = 500
        self.reset_dist = 3.0
        self.max_push_effort = 400.0

        self.cam_width = 512
        self.cam_height = 512
        
        # Set observation and action dimensions BEFORE calling super().__init__
        cfg["env"]["numObservations"] = 4
        cfg["env"]["numActions"] = 1
        
        # Call parent constructor
        super().__init__(
            config=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless
        )
        
        # Acquire state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # Wrap tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Reshape for easier access
        # dof_states shape is (num_envs * num_dofs * 2,) where 2 is [pos, vel]
        # We have 2 DOFs per env (cart slider, pole hinge)
        num_dofs = 2
        self.dof_states = self.dof_states.view(self.num_envs, num_dofs,  -1)
        self.dof_pos = self.dof_states[..., 0]  # positions
        self.dof_vel = self.dof_states[..., 1]  # velocities
        
        print(f"Cartpole initialized: {self.num_envs} envs on {self.device}")

    def create_sim(self):
        """Create the simulation."""
        self.sim = super().create_sim(
            self.device_id, 
            self.graphics_device_id, 
            self.physics_engine, 
            self.sim_params
        )

        # add light source
        


        self._create_ground_plane()
        self._create_envs()

    def _create_ground_plane(self):
        """Add ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """Create environments and actors."""
        # Load asset
        asset = self._load_cartpole_asset()
        
        # Environment setup
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.cartpole_handles = []
        self.envs = []
        self.cam_handles = []
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.cam_width
        cam_props.height = self.cam_height
        cam_props.enable_tensors = True  # allows rendering to a PyTorch tensor
        
        # Create environments
        for i in range(self.num_envs):
           
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env_ptr)
            
            # Add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            handle = self.gym.create_actor(env_ptr, asset, pose, f"cartpole_{i}", i, 1)
            cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)

            cam_pos = gymapi.Vec3(0.5, 0.0, 1.0)  # x, y, z


            cam_transform = gymapi.Transform()
            cam_transform.p = gymapi.Vec3(1.0, 0.0, 1.0)  # in front of the cart
            cam_transform.r = gymapi.Quat(0.5, 0.5, 0.5, 0.5)  # identity

            self.gym.set_camera_transform(cam_handle, env_ptr, cam_transform)

            
            # Configure DOFs
            dof_props = self.gym.get_actor_dof_properties(env_ptr, handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, handle, dof_props)
            
            self.cartpole_handles.append(handle)
            self.cam_handles.append(cam_handle)

    def _load_cartpole_asset(self):
        
        """Load the cartpole URDF asset."""
        # Try multiple possible paths
        possible_paths = [
            ('assets', 'urdf/cartpole.urdf'),
            ('../../assets', 'urdf/cartpole.urdf'),
            ('.', 'cartpole.urdf'),
        ]
        
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        
        # Try each path
        for asset_root, asset_file in possible_paths:
            full_path = os.path.join(asset_root, asset_file)
            if os.path.exists(full_path):
                print(f"Loading asset from: {full_path}")
                try:
                    asset = self.gym.load_asset(self.sim, asset_root, asset_file, options)
                    return asset
                except Exception as e:
                    print(f"Failed to load from {full_path}: {e}")
                    continue
        
        # If no file found, create temporary URDF
        print("No URDF found, creating temporary file...")
        return self._create_temp_urdf_asset()

    def _create_temp_urdf_asset(self):
        """Create a temporary URDF file and load it."""
        import tempfile
        
        urdf_content = """<?xml version="1.0" ?>
<robot name="cartpole">
  <link name="cart">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
  </link>

  <link name="pole">
    <inertial>
      <origin xyz="0 0 0.5"/>
      <mass value="0.1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <cylinder radius="0.05" length="1.0"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <cylinder radius="0.05" length="1.0"/>
      </geometry>
    </collision>
  </link>

  <link name="world"/>

  <joint name="slider_to_cart" type="prismatic">
    <parent link="world"/>
    <child link="cart"/>
    <origin xyz="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit effort="1000.0" lower="-10" upper="10" velocity="100.0"/>
  </joint>

  <joint name="cart_to_pole" type="continuous">
    <parent link="cart"/>
    <child link="pole"/>
    <origin xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""
        
        # Create temp directory and file
        temp_dir = tempfile.mkdtemp()
        urdf_file = os.path.join(temp_dir, "cartpole.urdf")
        
        with open(urdf_file, 'w') as f:
            f.write(urdf_content)
        
        print(f"Created temporary URDF at: {urdf_file}")
        
        # Load asset
        options = gymapi.AssetOptions()
        options.fix_base_link = True
        
        asset = self.gym.load_asset(self.sim, temp_dir, "cartpole.urdf", options)
        return asset

    def pre_physics_step(self, actions: torch.Tensor):
        """Apply actions to the environment.
        
        Args:
            actions: Tensor of actions to apply
        """
        actions_tensor = actions.to(self.device)
        forces = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        forces[:, 0] = self.max_push_effort * actions_tensor.squeeze()
        
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    def post_physics_step(self):
        """Compute observations, rewards, and resets after physics step."""
        self.progress_buf += 1
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.gym.render_all_camera_sensors(self.sim)
        
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""
        self.obs_buf[:, 0] = self.dof_pos[:, 0]  # cart position
        self.obs_buf[:, 1] = self.dof_vel[:, 0]  # cart velocity
        self.obs_buf[:, 2] = self.dof_pos[:, 1]  # pole angle
        self.obs_buf[:, 3] = self.dof_vel[:, 1]  # pole angular velocity
        image = self.gym.get_camera_image(self.sim, self.envs[0], self.cam_handles[0], gymapi.IMAGE_COLOR).reshape(self.cam_width, self.cam_height, 4)
        # image = image.astype(np.uint8)
        # print(image.shape)
        cv2.imshow("Cartpole", image)
        cv2.waitKey(1)
        return self.obs_buf
    

    def compute_reward(self):
        """Compute rewards and resets."""
        cart_pos = self.dof_pos[:, 0]
        pole_angle = self.dof_pos[:, 1]
        
        # Calculate reward
        cart_penalty = torch.abs(cart_pos) * 0.01
        pole_penalty = torch.abs(pole_angle) * 2.0
        alive_reward = 1.0
        
        self.rew_buf[:] = alive_reward - cart_penalty - pole_penalty
        
        # Determine resets
        self.reset_buf = torch.where(
            torch.abs(cart_pos) > self.reset_dist,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
        
        self.reset_buf = torch.where(
            torch.abs(pole_angle) > np.pi / 2,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )
        
        self.reset_buf = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def reset_idx(self, env_ids):
        """Reset specific environments."""
        num_resets = len(env_ids)
        
        # Random initial states
        positions = 0.2 * (torch.rand((num_resets, 2), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((num_resets, 2), device=self.device) - 0.5)
        
        self.dof_pos[env_ids] = positions
        self.dof_vel[env_ids] = velocities
        
        # Reset buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # Apply to simulation
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )