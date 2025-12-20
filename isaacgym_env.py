from utils.parse_task import parse_task, load_task_config
import open3d as o3d
import numpy as np
import torch
from threading import Thread
import time

class IsaacGymEnv:
    def __init__(self, cfg, args, task_args):
        self.cfg = cfg
        self.cfg['env']['numEnvs'] = args.num_envs
        print("Initializing environment...")
        task = parse_task(task_args, cfg, cfg_train=None, sim_params=None)
        self.task = task
        self.workspace_bounds_max = self.task.workspace_bounds_max
        self.workspace_bounds_min = self.task.workspace_bounds_min
        print(f"\nEnvironment initialized successfully!")
        print(f"Observation space: {task.num_obs}")
        print(f"Action space: {task.num_acts}")
        print(f"Device: {task.device}")
        print("\nStarting simulation...\n")

        self.descriptions = np.array(['task_1', 'task_2s'])

        self.name2ids = {}
        self.ids2name = []
        for i, name in enumerate(self.task.object_names):
            self.ids2name.append(name)
            self.name2ids[name] = i
        
        self._reset_task_variables()

        self.running = True
        self.cur_action = None

        self.main_thread = Thread(target=self.run)
        self.main_thread.start()

    def get_object_names(self):
        return self.task.object_names  
    
    def load_task(self, task):
        print(f"Loading task: {task}")
        pass

    def get_3d_obs_by_name(self, query_name):
        assert query_name in self.task.object_names, f"{query_name} is not an object in the task"

        obj_id = self.name2ids[query_name]

        points, masks, normals = [], [], []
        # print(">>")
        for i, pcl in enumerate(self.task.vision_buffer):
            print(pcl)
    
    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        self.task.process_vision_buffer()
        # vision_buffer  = self.task.vision_buffer
        # pcl = vision_buffer['pcl']
        
    def reset(self):
        self.task.reset()
        zero_action = torch.zeros([self.task.num_envs, self.task.num_acts])
        self.task.step(zero_action)
        self.task.process_vision_buffer()
        obs = self.task.vlm_obs
        self.latest_obs = obs
        return self.descriptions, obs
    
    def apply_action(self, action):
        self.cur_action = action
        self.latest_action = action
        self.latest_obs = self.task.vlm_obs

        obs, reward, terminate, _ = self.step(action)
        obs = self._process_obs(obs)

        self.latest_obs = obs
        self.latest_reward = reward
        self.latest_terminate = terminate
        self.latest_action = action
        self.render()
        # grasped_objects = self.task.get_grasped_objects()
        # if len(grasped_objects) > 0:
        #     self.grasped_obj_ids = [obj.get_handle() for obj in grasped_objects]
        return obs, reward, terminate


    def step(self, action):
        # need to mdify a osc executer
        # self.cur_action = action
        return self.task.vlm_obs, self.task.rew_buf, self.task.dones, self.task.extras
    def render(self):
        self.task.render()

    def move_to_pose(self, pose, speed=None):
        """
        Moves the robot arm to a specific pose.

        Args:
            pose: The target pose.
            speed: The speed at which to move the arm. Currently not implemented.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([pose, [self.latest_action[-1]]])
        return self.apply_action(action)
    
    def open_gripper(self):
        """
        Opens the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [1.0]])
        return self.apply_action(action)

    def close_gripper(self):
        """
        Closes the gripper of the robot.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [-1.0]])
        return self.apply_action(action)

    def set_gripper_state(self, gripper_state):
        """
        Sets the state of the gripper.

        Args:
            gripper_state: The target state for the gripper.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        action = np.concatenate([self.latest_obs.gripper_pose, [gripper_state]])
        return self.apply_action(action)

    def reset_to_default_pose(self):
        """
        Resets the robot arm to its default pose.

        Returns:
            tuple: A tuple containing the latest observations, reward, and termination flag.
        """
        if self.latest_action is None:
            action = np.concatenate([self.init_obs.gripper_pose, [self.init_obs.gripper_open]])
        else:
            action = np.concatenate([self.init_obs.gripper_pose, [self.latest_action[-1]]])
        return self.apply_action(action)

    def get_ee_pose(self):
        assert self.latest_obs is not None, "Please reset the environment first"
        return self.latest_obs.gripper_pose

    def get_ee_pos(self):
        pose = self.get_ee_pose()
        return pose[:3], pose[7:10]

    def get_ee_quat(self):
        pose = self.get_ee_pose()
        return pose[3:7], pose[11:]

    def get_last_gripper_action(self):
        """
        Returns the last gripper action.

        Returns:
            float: The last gripper action.
        """
        if self.latest_action is not None:
            return self.latest_action[-1]
        else:
            return self.init_obs.gripper_open

    def _reset_task_variables(self):
        """
        Resets variables related to the current task in the environment.

        Note: This function is generally called internally.
        """
        self.init_obs = None
        self.latest_obs = None
        self.latest_reward = None
        self.latest_terminate = None
        self.latest_action = None
        self.grasped_obj_ids = None
        # scene-specific helper variables
        self.arm_mask_ids = None
        self.gripper_mask_ids = None
        self.robot_mask_ids = None
        self.obj_mask_ids = None
        self.name2ids = {}  # first_generation name -> list of ids of the tree
        self.id2name = {} 

    
    
    def _process_obs(self, obs):
        return obs

    def _process_action(self, input_action):
        """
        根据目标位姿计算OSC控制的action
        
        参数:
            tar_pose: 目标位姿 [num_envs, 7] 或 [7]
                    格式: [x, y, z, qx, qy, qz, qw]
        
        返回:
            action: [num_envs, 6] 或 [6]
                格式: [dx, dy, dz, droll, dpitch, dyaw] (轴角表示)
        """

        input_action = input_action.reshape([-1, 8])

        if input_action.shape[0] > self.task.num_robots_per_env:
            print("Warning: action cmd exceed the number of robot!")
            return None
        
        output_action = torch.zeros([self.task.num_envs, 7 * self.task.num_robots_per_env])

        for i in range(input_action.shape[0]): 
            tar_pose = input_action[i][:7]
            gripper_action = input_action[i][7]
            # 确保tar_pose是tensor
            if isinstance(tar_pose, np.ndarray):
                tar_pose = torch.from_numpy(tar_pose).float().to(self.task.rl_device)
            elif not isinstance(tar_pose, torch.Tensor):
                tar_pose = torch.tensor(tar_pose, dtype=torch.float32, device=self.task.rl_device)
            
            
            eef_pose = torch.tensor(self.get_ee_pose(),
                                    device=self.task.rl_device)[i]
            
            tar_pos = tar_pose[:3]  # [3]
            tar_quat = tar_pose[3:7]  # [4] (x, y, z, w)
            
            eef_pos = eef_pose[:3]  # [3]
            eef_quat = eef_pose[3:7]  # [4] (x, y, z, w)
            
            pos_delta = tar_pos - eef_pos  # [3]

            # pos_err = np.linalg.norm(pos_delta)
            
            eef_quat_conj = quat_conjugate(eef_quat)

            tar_quat = tar_quat.unsqueeze(0)
            eef_quat_conj = eef_quat_conj.unsqueeze(0)
            delta_quat = quat_mul(tar_quat, eef_quat_conj)

            # delta_err = quat_distance(delta_quat, np.array([1, 0, 0, 0]))
            
            rot_delta = quat_to_axis_angle(delta_quat)  # [3]

            rot_delta = rot_delta.squeeze(0)
            
            action = torch.cat([pos_delta, rot_delta], dim=-1)  # [6]
            
            action[:3] = torch.clamp(action[:3], -0.1, 0.1)
            # rot_delta可以限制在±0.5rad
            action[3:6] = torch.clamp(action[3:6], -0.5, 0.5)

            output_action[0, i*7:i*7+6] = action
            output_action[0, i*7+6] = gripper_action

        return output_action
      
    def run(self):
        while self.running:
            
            if self.cur_action is None:
                time.sleep(0.1)
                continue
            # print(">>>>>>")
            action = self._process_action(self.cur_action)
            if action is None:
                continue
            # print(action[0, -1])
            self.task.step(action)
            self.render()
    
    def close(self):
        self.running = False
        self.main_thread.join()
        

def quat_distance(q_1, q_2):
    # type: (Tensor, Tensor) -> Tensor
    q_rot = torch.zeros_like(q_1, device=q_1.device, dtype=torch.float32)
    q_rot[:, 1] = 1
    q_1 = quat_mul(q_rot, q_1)
    ##
    dot_quat = torch.sum(q_1 * q_2, dim=-1)
    return 2 * torch.acos(torch.clamp(dot_quat, -1.0 + 1e-7, 1.0 - 1e-7))

def quat_mul(q1, q2):
    """四元数乘法 (x, y, z, w)"""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([x, y, z, w], dim=-1)

def quat_conjugate(q):
    """四元数共轭 (x, y, z, w) -> (-x, -y, -z, w)"""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

def quat_to_axis_angle(quat):
    """
    将四元数转换为轴角表示
    输入: quat [x, y, z, w]
    输出: axis_angle [ax, ay, az] 其中模长为旋转角度
    """
    # 归一化四元数
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # 计算旋转角度
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
    
    # 计算旋转轴
    sin_half_angle = torch.sin(angle / 2.0)
    
    # 避免除以零
    axis = torch.zeros_like(quat[..., :3])
    mask = sin_half_angle.abs() > 1e-6
    
    if mask.any():
        axis[mask] = torch.stack([x[mask], y[mask], z[mask]], dim=-1) / sin_half_angle[mask].unsqueeze(-1)
        axis[mask] = axis[mask] * angle[mask].unsqueeze(-1)
    
    return axis


