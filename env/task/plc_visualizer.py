import open3d as o3d
import numpy as np

class PointCloudVisualizer:
    def __init__(self, name):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=f'Dynamic Point Cloud: {name}', width=640, height=640)
        
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        self.vis.add_geometry(self.coordinate_frame)
        
        self.is_first_update = True
    
    def update(self, pcl):
        """更新点云显示"""
        self.pcd.points = pcl.points
        self.pcd.colors = pcl.colors
        if pcl.has_normals():
            self.pcd.normals = pcl.normals
        
        if self.is_first_update:
            self.vis.reset_view_point(True)
            self.is_first_update = False
        
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def close(self):
        """关闭可视化窗口"""
        self.vis.destroy_window()