import numpy as np

cam_width = 512
cam_height = 512            


class VLM_Observation():
    def __init__(self, num_robot = 1, num_cam = 2):
        self.cam_rgb = np.zeros([num_cam, cam_width, cam_height, 3])  
        self.cam_depth = np.zeros([num_cam, cam_width, cam_height])      
        self.cam_mask = np.zeros([num_cam, cam_width, cam_height])  
        self.cam_point_cloud = []
        for i in range(num_cam):
            self.cam_point_cloud.append([])                                                                                                                                                                                                                                                                                                                                                                  
     
        self.joint_velocities = np.zeros([num_robot, 7])
        self.joint_positions = np.zeros([num_robot, 7])
        self.joint_forces = np.zeros([num_robot, 7])

        self.gripper_open = np.zeros([num_robot, 2])
        self.gripper_pose = np.zeros([num_robot, 7])
        self.gripper_matrix = None
        self.gripper_touch_forces = None
        self.task_low_dim_state = None

        self.misc = {}

