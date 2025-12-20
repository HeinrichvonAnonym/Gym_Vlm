import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp
from env.task.vec_task import VecTask

from gym import spaces
import yaml
import open3d as o3d

import cv2

from env.task.vlm_observation import VLM_Observation

from env.task.plc_visualizer import PointCloudVisualizer

from scipy.spatial import transform as trans
from scipy.spatial.transform import Rotation as R

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import torch.nn.functional as F
from torchvision.transforms.functional import resize, InterpolationMode



@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class MultiFranka(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        self.cfg = cfg
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.workspace_bounds_min = self.cfg["env"]["workspaceBoundsMin"]
        self.workspace_bounds_max = self.cfg["env"]["workspaceBoundsMax"]
        self.object_names = self.cfg["env"]["objectNames"]

        self.num_robots_per_env = self.cfg["env"]["numRobotsPerEnv"]
        self.num_cams_per_env = self.cfg["env"]["numCamsPerEnv"]

        self.table_parmas = self.cfg["env"]["table"]

        self.cam_poses = self.cfg["env"]["camPoses"]
        self.cam_height = self.cfg["env"]["camHeight"]
        self.cam_width = self.cfg["env"]["camWidth"]

        self.robot_offset = self.cfg["env"]["robotOffset"]

        self.cam_rots = self.cfg["env"]["camRots"]

        self.cam_intrinsics = self.cfg["env"]["camIntrinsics"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 9 if self.control_type == "osc" else 9
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 * self.num_robots_per_env if self.control_type == "osc" else 8 * self.num_robots_per_env

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        # pcl settings
        self.pcl_transforms = []  # 用于存储每个相机的完整变换矩阵
        self.pcl_visualizers = []
        for i in range(self.num_cams_per_env):
            self.pcl_visualizers.append(PointCloudVisualizer(f"pcl_{i}"))        
            cam_pose = self.cam_poses[i]
            position = np.array(cam_pose[:3])
            # rotation_mat = R.from_euler('ZYX', euler_zyx, degrees=False).as_matrix()
            pcl_transform = np.eye(4)
            pcl_transform[:3, :3] = self.cam_rots[i]
            pcl_transform[:3, 3] = position        
            self.pcl_transforms.append(pcl_transform)
        
        self.init_sam_model()

    def init_sam_model(self):
        """在DualFranka.__init__中调用此方法"""
        # 加载模型
        self.sam_model = sam_model_registry["vit_b"](checkpoint="models/sam_vit_b_01ec64.pth")
        
        # 将模型移到GPU
        self.sam_model.to(device="cuda:0")
        
        # 创建mask生成器,使用合理的参数
        self.sam_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,  # 减少采样点以加快速度
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # 过滤小掩码
        )
        
        print(f"SAM model initialized on {self.device}")
        
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def load_table_asset(self):
        # Create table asset
        self.table_pos = [0.0, 0.0, 1.0]
        self.table_thickness = 0.05
        self.table_opts = gymapi.AssetOptions()
        self.table_opts.fix_base_link = True
        table_length = self.table_parmas["size"][0]
        table_width = self.table_parmas["size"][1]
        self.table_asset = self.gym.create_box(self.sim, *[table_length, table_width, self.table_thickness], self.table_opts)
    
    def load_stand_asset(self):
        # Create robot stand asset
        self.table_stand_height = 0.05
        self.table_stand_pos = [-0.25, 0.0, 1.0 + self.table_thickness / 2 + self.table_stand_height / 2]
        self.table_stand_opts = gymapi.AssetOptions()
        self.table_stand_opts.fix_base_link = True
        self.table_stand_asset = self.gym.create_box(self.sim, *[0.3, 0.3, self.table_stand_height], self.table_opts)

    
    def load_robot_asset(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "robots/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.flip_visual_attachments = True
        self.asset_options.fix_base_link = True
        self.asset_options.collapse_fixed_joints = False
        self.asset_options.disable_gravity = True
        self.asset_options.thickness = 0.001
        self.asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        self.asset_options.use_mesh_materials = True
        self.franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, self.asset_options)

        

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(self.franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(self.franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            self.franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                self.franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                self.franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                self.franka_dof_props['stiffness'][i] = 7000.0
                self.franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(self.franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(self.franka_dof_props['upper'][i])
            self._franka_effort_limits.append(self.franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        self.franka_dof_props['effort'][7] = 200
        self.franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(-0.25, 0.0, 1.0 + self.table_thickness / 2 + self.table_stand_height)
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        self.table_start_pose = gymapi.Transform()
        self.table_start_pose.p = gymapi.Vec3(*self.table_pos)
        self.table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table stand
        self.table_stand_start_pose = gymapi.Transform()
        self.table_stand_start_pose.p = gymapi.Vec3(*self.table_stand_pos)
        self.table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(self.franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(self.franka_asset)
        self.max_agg_bodies = num_franka_bodies + 2     # 1 for table, table stand
        self.max_agg_shapes = num_franka_shapes + 2     # 1 for table, table stand

    def load_humanoid(self):
        human_asset_root = "assets/human"
        human_asset_file = "smplx_humanoid.xml"
        human_asset_options = gymapi.AssetOptions()
        human_asset_options.flip_visual_attachments = True
        human_asset_options.fix_base_link = False # 固定基座
        human_asset_options.collapse_fixed_joints = True  # 合并固定关节
        human_asset_options.disable_gravity = False  # 禁用重力
        human_asset_options.thickness = 0.001

        # Define start psoe for humanoid
        self.human_start_pose = gymapi.Transform()
        self.human_start_pose.p = gymapi.Vec3(*self.table_stand_pos)
        self.human_start_pose.p.z = 1.2
        self.human_start_pose.p.x = 0
        self.human_start_pose.r = gymapi.Quat(0.0, 0.0, 0.707, 0.707)
        
        # 关键：设置为NONE模式，不施加任何驱动力
        human_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        # 使用mesh材质
        human_asset_options.use_mesh_materials = True
        
        human_asset = self.gym.load_asset(self.sim, human_asset_root, human_asset_file, human_asset_options)
        self.human_asset = human_asset

    def create_robot(self, env_ptr, index):
        for i in range(self.num_robots_per_env):
            """
                add franka offset here
            """
            offset = self.robot_offset[i]
            pose = gymapi.Transform()
            pose.p.x = self.franka_start_pose.p.x + offset[0]
            pose.p.y = self.franka_start_pose.p.y + offset[1]
            pose.p.z = self.franka_start_pose.p.z + offset[2]

            pose.r.x = offset[3]
            pose.r.y = offset[4]
            pose.r.z = offset[5]
            pose.r.w = offset[6]

            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                    1.0 + self.table_thickness / 2 + self.table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                pose.r = gymapi.Quat(*new_quat)

            franka_actor = self.gym.create_actor(env_ptr, self.franka_asset, pose, f"franka_{i+1}", index, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, self.franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)
            
            self.frankas.append(franka_actor)
    
    def create_table(self, env_ptr, index):
        table_actor = self.gym.create_actor(env_ptr, self.table_asset, self.table_start_pose, "table", index, 0, 0)
        self.num_stable_aware_per_env += 1
    
    def create_stand(self, env_ptr, index):
        for i in range(self.num_robots_per_env):
            offset = self.robot_offset[i]

            pose = gymapi.Transform()
            pose.p = self.table_stand_start_pose.p
            pose.p.x += offset[0]
            pose.p.y += offset[1]
            pose.p.z += offset[2]

            pose.r.x = offset[3]
            pose.r.y = offset[4]
            pose.r.z = offset[5]
            pose.r.w = offset[6]

            table_stand_actor = self.gym.create_actor(env_ptr, self.table_stand_asset, pose, f"table_stand_{i}",
                                                    index, 1, 0)
        
            self.num_stable_aware_per_env += 1
        
    def create_human(self, env_ptr, index):
        humanoid_actor = self.gym.create_actor(env_ptr, self.human_asset, self.human_start_pose, "human", index, 0,  0)
            
        # **关键：设置人形的DOF属性，确保所有关节都是被动的**
        self.human_dof_props = self.gym.get_actor_dof_properties(env_ptr, humanoid_actor)
        num_human_dofs = self.gym.get_actor_dof_count(env_ptr, humanoid_actor)
        
        for j in range(num_human_dofs):
            # 设置为NONE模式（无驱动）
            self.human_dof_props['driveMode'][j] = gymapi.DOF_MODE_NONE
            
            # 设置关节属性为柔顺（低刚度，高阻尼）
            self.human_dof_props['stiffness'][j] = 20  
            self.human_dof_props['damping'][j] = 10.0   # 适度阻尼，避免抖动
            self.human_dof_props['friction'][j] = 1.0   # 添加摩擦力
            self.human_dof_props['effort'][j] = 0.0     # 无输出力
            
        # 应用DOF属性
        self.gym.set_actor_dof_properties(env_ptr, humanoid_actor, self.human_dof_props)
        
        human_dof_states = np.zeros(num_human_dofs, dtype=gymapi.DofState.dtype)

        # human_dof_states['pos'] = [0.0] * num_human_dofs  # 根据实际调整
        human_dof_states['vel'] = 0.0  # 零速度
        
        self.gym.set_actor_dof_states(
            env_ptr, 
            humanoid_actor, 
            human_dof_states, 
            gymapi.STATE_ALL
        )
        # **可选：设置人形刚体属性，使其质量分布更真实**
        human_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, humanoid_actor)
        for body_idx in range(len(human_body_props)):
            # 降低质量可以减少动力学影响
            human_body_props[body_idx].mass *= 0.5
        self.gym.set_actor_rigid_body_properties(env_ptr, humanoid_actor, human_body_props)
        
        # 存储人形句柄（如果需要后续访问）
        if not hasattr(self, 'humanoids'):
            self.humanoids = []
        self.humanoids.append(humanoid_actor)

        self.num_humanoid_per_env += 1
    
    def create_cameras(self, env_ptr, index):
        for j in range(self.num_cams_per_env):
            cam_props = gymapi.CameraProperties()
            cam_props.width = self.cam_width
            cam_props.height = self.cam_height
            cam_props.enable_tensors = True
            horizontal_fov = self.cam_intrinsics[j]["horizontal_fov"]

            cam_props.horizontal_fov = horizontal_fov
        
            fx = self.cam_width / (2.0 * np.tan(np.pi * horizontal_fov / 360))
            fy = fx * self.cam_height / self.cam_width

            if "cx" not in self.cam_intrinsics[j].keys():
                self.cam_intrinsics[j]["cx"] = self.cam_width / 2
                self.cam_intrinsics[j]["cy"] = self.cam_height / 2

            self.cam_intrinsics[j]["fx"] = fx
            self.cam_intrinsics[j]["fy"] = fy

            cam_pose = self.cam_poses[j]
            cam_transform = gymapi.Transform()
            cam_transform.p = gymapi.Vec3(cam_pose[0], cam_pose[1], cam_pose[2])
            cam_transform.r = gymapi.Quat.from_euler_zyx(cam_pose[3], cam_pose[4], cam_pose[5])

            cam = self.gym.create_camera_sensor(env_ptr, cam_props)
            self.gym.set_camera_transform(cam, env_ptr, cam_transform)

            self.cameras.append(cam)

    def _create_envs(self, num_envs, spacing, num_per_row):
        x_low = self.workspace_bounds_min[0]
        x_high = self.workspace_bounds_max[0]
        y_low = self.workspace_bounds_min[1]
        y_high = self.workspace_bounds_max[1]
        z_low = self.workspace_bounds_min[2]
        z_high = self.workspace_bounds_max[2]

        self.lower = gymapi.Vec3(float(x_low), float(y_low), float(z_low))
        self.upper = gymapi.Vec3(float(x_high), float(y_high), float(z_high))

        self.num_stable_aware_per_env  = 0
        self.num_humanoid_per_env = 0

        self.load_table_asset()
        self.load_stand_asset()
        self.load_robot_asset()
        self.load_humanoid()

        self.humanoids = []
        self.frankas = []
        self.envs = []
        self.cameras = []
        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, self.lower, self.upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, self.max_agg_bodies, self.max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            self.create_robot(env_ptr, i)
            self.create_stand(env_ptr, i)

            # Create table
            # remember add 1 stable aware per env
            self.create_table(env_ptr, i)
             
            # # Create Humanoid #####
            self.create_human(env_ptr, i)

            # create cameras
            self.create_cameras(env_ptr, i)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr) 
        
        # Setup data
        self.init_data()

    def init_basical_tensors(self):
        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[:,:self.num_robots_per_env * 9, 0]
        self._qd = self._dof_state[:, :self.num_robots_per_env * 9, 1]

    
    def init_param_tensors(self, env_ptr):
        for franka_handle in range(self.num_robots_per_env):
            self.handles.append({
                # Franka
                "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
                "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
                "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
                "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
                "panda_hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            })

            self._eef_states.append(self._rigid_body_state[:, self.handles[franka_handle]["grip_site"], :])
            self._eef_lf_states.append(self._rigid_body_state[:, self.handles[franka_handle]["leftfinger_tip"], :])
            self._eef_rf_states.append(self._rigid_body_state[:, self.handles[franka_handle]["rightfinger_tip"], :])
            self._eef_pre_states.append(self._rigid_body_state[:, self.handles[franka_handle]["panda_hand"], :])

        

            self._jacobians.append(self.gym.acquire_jacobian_tensor(self.sim, f"franka_{franka_handle+1}"))
            self.jacobians.append(gymtorch.wrap_tensor(self._jacobians[franka_handle]))
            hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, 0)['panda_hand_joint'] # use first index
            self._j_eefs.append(self.jacobians[franka_handle][:, hand_joint_index, :, :7])
            self._massmatrixs.append(self.gym.acquire_mass_matrix_tensor(self.sim, f"franka_{franka_handle+1}"))
            self.mms.append(gymtorch.wrap_tensor(self._massmatrixs[franka_handle]))
            self._mms.append(self.mms[franka_handle][:, :7, :7])
    
    def init_control_tensors(self):
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        for i in range(self.num_robots_per_env):
            start_dof = i * 9
            arm_control = self._effort_control[:, start_dof:start_dof+7]
            gripper_control = self._pos_control[:, start_dof+7:start_dof+9]
            self._arm_controls.append(arm_control)
            self._gripper_controls.append(gripper_control)


    def init_data(self):
        # Setup sim handles
        # init vlm observation
        self.vlm_obs = VLM_Observation(num_robot=self.num_robots_per_env,
                                       num_cam=self.num_cams_per_env)
        self.dones = np.zeros(self.num_envs, dtype=np.int32)
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # initialize basical tensor buffer
        self.init_basical_tensors()

        # initialize params buffer
        env_ptr = self.envs[0]
        self.handles = []
        self._jacobians = []
        self.jacobians = []
        self._j_eefs = []
        self._massmatrixs = []
        self._mms = []
        self.mms = []
        self._eef_states = []
        self._eef_lf_states = []
        self._eef_rf_states = []
        self._eef_pre_states = []
        self.init_param_tensors(env_ptr)

        # Initialize control
        self._arm_controls = []
        self._gripper_controls = []
        self.init_control_tensors()

        # Initialize indices
        actors_per_env = int(self.num_robots_per_env + self.num_stable_aware_per_env + self.num_humanoid_per_env)
        self._global_indices = torch.arange(self.num_envs * actors_per_env, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_states[0][:, :3],
            "eef_quat": self._eef_states[0][:, 3:7],
            "eef_vel": self._eef_states[0][:, 7:],
            "eef_lf_pos": self._eef_lf_states[0][:, :3],
            "eef_rf_pos": self._eef_rf_states[0][:, :3],
            # "pre_eef_quat": self._eef_pre_state[:, 3:7],
        })
        for idx in range(self.num_robots_per_env):
            self._process_robot_buffer(idx)

    def _process_robot_buffer(self, robot_idx):
        start_dof = robot_idx * 9
        end_dof = (robot_idx + 1 ) * 9
        self.vlm_obs.joint_positions[robot_idx] = self._q[0, start_dof:start_dof + 7].cpu().numpy()
        self.vlm_obs.joint_velocities[robot_idx] = self._qd[0, start_dof:start_dof + 7].cpu().numpy()

        self.vlm_obs.gripper_open[robot_idx] = self._q[0, start_dof+7:end_dof].cpu().numpy()
        self.vlm_obs.gripper_pose[robot_idx] = self._eef_states[robot_idx][0, :7].cpu().numpy()
        # print(self.vlm_obs.gripper_pose[robot_idx])

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self._update_states()

    def compute_reward(self, actions):
        # Return zero reward
        self.rew_buf[:] = 0.0
        # No automatic reset
        self.reset_buf[:] = 0

    def compute_observations(self):
        self._refresh()
        self._update_states()
        obs = ["eef_pos", "q_gripper"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        for i in range(self.num_robots_per_env):
            start_dof = i * 9
            end_dof = (i + 1) * 9
            self._q[env_ids, start_dof:end_dof] = pos
            self._qd[env_ids, start_dof:end_dof] = torch.zeros_like(self._qd[env_ids, start_dof:end_dof])

            # Set any position control to the current position, and any vel / effort control to be 0
            # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
            self._pos_control[env_ids, start_dof:end_dof] = pos
            self._effort_control[env_ids, start_dof:end_dof] = torch.zeros_like(pos)

            # Deploy updates
            multi_env_ids_int32 = self._global_indices[env_ids, i].flatten()
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self._pos_control),
                                                            gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                            len(multi_env_ids_int32))
            self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self._effort_control),
                                                            gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                            len(multi_env_ids_int32))
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self._dof_state),
                                                gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _compute_osc_torques(self, dpose, robot_idx):
        start_dof = 9*robot_idx
        q, qd = self._q[:, start_dof:start_dof+7], self._qd[:, start_dof:start_dof+7]
        mm_inv = torch.inverse(self._mms[robot_idx])
        m_eef_inv = self._j_eefs[robot_idx] @ mm_inv @ torch.transpose(self._j_eefs[robot_idx], 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eefs[robot_idx], 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self._eef_states[robot_idx][:, 7:]).unsqueeze(-1)

        j_eef_inv = m_eef @ self._j_eefs[robot_idx] @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mms[robot_idx] @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eefs[robot_idx], 1, 2) @ j_eef_inv) @ u_null

        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))
        return u

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        # print(actions[0])
        for i in range(self.num_robots_per_env):
            start_dof = i * 9
            end_dof = (i+1) * 9
            u_arm, u_gripper = self.actions[:, i * 7 : i * 7 + 6], self.actions[:, i * 7 + 6]

            # print(u_arm, u_gripper)
            # print(self.cmd_limit, self.action_scale)

            # Control arm (scale value first)
            # print(u_arm.shape)
            u_arm = u_arm * self.cmd_limit / self.action_scale
            if self.control_type == "osc":
                u_arm = self._compute_osc_torques(dpose=u_arm, robot_idx=i)

            # print(u_arm.shape)
            self._arm_controls[i][:, :] = u_arm

            # Control gripper
            u_fingers = torch.zeros_like(self._gripper_controls[i])
            u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                        self.franka_dof_lower_limits[-2].item())
            u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                        self.franka_dof_lower_limits[-1].item())
            # Write gripper command to appropriate tensor buffer
            self._gripper_controls[i][:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def _depth_to_heatmap(self, depth_image, near_plane=0.1, far_plane=3.0):
        """Convert depth to heatmap with fixed range."""
        depth_valid = depth_image.copy()
        depth_valid[depth_valid < near_plane] = near_plane
        depth_valid[depth_valid > far_plane] = far_plane
        depth_valid[depth_valid <= 0] = far_plane
        depth_normalized = ((depth_valid - near_plane) / (far_plane - near_plane) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return heatmap

    def process_vision_buffer(self, vis=True):
        self.gym.render_all_camera_sensors(self.sim)
        self.vision_buffer = {}
        self.points = []
        self.normals = []
        self.colors = []
        self.cam_masks = []

        for cam_idx in range(self.num_cams_per_env):
            self.process_cam_obs(cam_idx, vis=vis)

    def process_cam_obs(self, idx, vis=True):
        rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.cameras[idx], gymapi.IMAGE_COLOR)
        rgb_image = rgb_image.reshape(self.cam_height, self.cam_width, 4)
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2BGR)

        cam_masks = self.get_obj_masks(rgb_image)
        self.cam_masks.append(cam_masks)

        depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.cameras[idx], gymapi.IMAGE_DEPTH)
        depth_image = depth_image.reshape(self.cam_height, self.cam_width) * -1

        transform = self.pcl_transforms[idx]
        pcl = get_pcl(rgb_bgr, depth_image,
                      cam_transform=transform,
                      intrinsic=self.cam_intrinsics[idx]
                      )
        points = pcl.points
        normals = pcl.normals
        colors = np.array(pcl.colors).reshape(-1, 3)
        points_arr = np.array(points).reshape(-1, 3)
        normals_arr = np.array(normals).reshape(-1, 3)

        self.vlm_obs.cam_rgb[idx] = rgb_image[:, :, :3]
        self.vlm_obs.cam_depth[idx] = depth_image
        self.vlm_obs.cam_point_cloud[idx] = points_arr

        if vis:
            depth_heatmap = self._depth_to_heatmap(depth_image)
            cv2.imshow(f"RGB View{idx}", rgb_bgr)
            cv2.imshow(f"Depth Heatmap{idx}", depth_heatmap)
            cv2.waitKey(1)
            self.pcl_visualizers[idx].update(pcl)
            if cam_masks:
                mask_visualization = visualize_masks_opencv(cam_masks, rgb_image, idx)
                cv2.imshow(f"SAM Masks Camera {idx}", mask_visualization)

    def get_obj_masks(self, rgb_raw: np.ndarray):
        """
        使用SAM生成对象掩码
        
        Args:
            rgb_raw: [H, W, 4] 或 [H, W, 3] 的numpy数组
        
        Returns:
            masks: SAM生成的掩码列表
        """
        # 确保是RGB三通道
        if rgb_raw.shape[2] == 4:
            rgb_np = rgb_raw[:, :, :3]
        else:
            rgb_np = rgb_raw.copy()
        
        # SAM需要uint8格式的[H, W, 3]图像
        if rgb_np.dtype != np.uint8:
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)
        
        try:
            # 直接传入numpy数组,SAM会自动处理
            print(f"Generating masks for image shape: {rgb_np.shape}")
            masks = self.sam_generator.generate(rgb_np)
            print(f"Generated {len(masks)} masks")
            return masks
        
        except Exception as e:
            print(f"Error generating masks: {e}")
            return []

    # def process_rgb(self, rgb, obj_name):
    #     mask = np.zeros_like(rgb[:, :, 0])
    #     mask[:, :] = 1
    #     return mask


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def quat_distance(q_1, q_2):
    # type: (Tensor, Tensor) -> Tensor
    q_rot = torch.zeros_like(q_1, device=q_1.device, dtype=torch.float32)
    q_rot[:, 1] = 1
    q_1 = quat_mul(q_rot, q_1)

    ##
    dot_quat = torch.sum(q_1 * q_2, dim=-1)
    ##

    dot_clamped = torch.clamp(dot_quat, min=-1.0, max=1.0)
    theta = 2 * torch.acos(torch.abs(dot_clamped))
    return 1 - theta / torch.pi

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]
    cube_a_quat = states["cubeA_quat"]
    cube_a_quat = torch.nn.functional.normalize(cube_a_quat, dim=-1)
   # 计算 EEF z 轴方向与竖直向下方向的一致性
    eef_quat = states["pre_eef_quat"]  # shape [num_envs, 4]
    eef_quat = torch.nn.functional.normalize(eef_quat, dim=-1)

    quat_reward = quat_distance(eef_quat, cube_a_quat)

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    quat_reward = torch.where(d < 0.15,  quat_reward, torch.zeros_like(quat_reward))

    rewards = reward_settings["r_dist_scale"] * dist_reward + 5.0 * quat_reward


    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf


def get_pcl(rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray=None,
            cam_transform: dict = None, intrinsic:dict=None):
    """创建点云，可选地应用相机变换"""
    if len(rgb.shape) == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]

    if mask is not None:
        mask = np.zeros_like(depth)

    depth_masked = depth.copy().astype(np.float32)
    depth_masked[mask == 0] = 0

    valid_depth_count = np.sum(depth_masked > 0)

    if valid_depth_count == 0:
        return o3d.geometry.PointCloud()

    rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_masked.astype(np.float32))

    # height, width = depth.shape
    # fov_rad = 69.4 * np.pi / 180
    # fx = fy = width / (2 * np.tan(fov_rad / 2))
    height=intrinsic['height']
    width = intrinsic['width']
    fx = intrinsic['fx']
    fy = intrinsic['fy']
    cx = intrinsic['cx']
    cy = intrinsic["cy"]


    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=fx, fy=fy,
        cx=cx, cy=cy
    )

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=10.0,
        convert_rgb_to_intensity=False
    )

    pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    if len(pcl.points) == 0:
        return pcl

    # pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    if cam_transform is not None:
        pcl.transform(cam_transform)

    if len(pcl.points) > 10:
        pcl.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    return pcl

def visualize_masks_opencv(masks, rgb_image, idx):
    """
    使用OpenCV显示SAM生成的掩码（适合你的现有可视化流程）
    """
    import cv2
    import numpy as np
    
    if not masks:
        print(f"Camera {idx}: No masks generated")
        return rgb_image[:, :, :3]  # 返回原始图像
    
    # 创建一个彩色掩码可视化
    mask_overlay = np.zeros_like(rgb_image[:, :, :3])
    
    # 为每个掩码分配不同的颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 黄色
    ]
    
    # 绘制所有掩码
    for i, mask_data in enumerate(masks[:6]):  # 最多显示6个掩码
        mask = mask_data['segmentation']
        color = colors[i % len(colors)]
        mask_overlay[mask] = color
    
    # 将掩码叠加到原始图像上（50%透明度）
    alpha = 0.5
    result = cv2.addWeighted(rgb_image[:, :, :3], 1 - alpha, mask_overlay, alpha, 0)
    
    # 在图像上添加文本信息
    cv2.putText(result, f'Masks: {len(masks)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示得分最高的掩码信息
    if masks:
        best_mask = max(masks, key=lambda x: x['predicted_iou'])
        cv2.putText(result, f'Best IOU: {best_mask["predicted_iou"]:.2f}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return result