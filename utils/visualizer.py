import open3d as o3d
import numpy as np
import cv2
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

def visualize_masks_opencv(masks, rgb_image, idx):
    if not masks:
        print(f"Camera {idx}: No masks generated")
        return rgb_image[:, :, :3]  # 返回原始图像
    
    # 创建一个彩色掩码可视化
    mask_overlay = np.zeros_like(rgb_image[:, :, :3], dtype=np.uint8)
    
    # 为每个掩码分配不同的颜色
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
    ]
    
    # 绘制所有掩码 - 关键修改：逐个叠加而不是替换
    for i, mask in enumerate(masks):  # 最多显示6个掩码
        color = np.array(colors[i % len(colors)], dtype=np.uint8)
        
        # 创建单个mask的彩色版本
        colored_mask = np.zeros_like(rgb_image[:, :, :3], dtype=np.uint8)
        colored_mask[mask] = color
        
        # 叠加到总的mask_overlay上（而不是直接赋值）
        mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.5, 0)
    
    # 将掩码叠加到原始图像上
    alpha = 0.5
    result = cv2.addWeighted(rgb_image[:, :, :3].astype(np.uint8), 1 - alpha, 
                            mask_overlay, alpha, 0)
    
    # 在图像上添加文本信息
    cv2.putText(result, f'Masks: {len(masks)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示得分最高的掩码信息
    # if masks:
    #     best_mask = max(masks, key=lambda x: x['predicted_iou'])
    #     cv2.putText(result, f'Best IOU: {best_mask["predicted_iou"]:.2f}', (10, 70), 
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return result