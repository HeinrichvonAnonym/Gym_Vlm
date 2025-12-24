from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
from groundingdino.util.inference import Model as GroundingDINOModel
import numpy as np
import torch
import cv2
from PIL import Image
from typing import List, Dict, Tuple


class SamInterface:
    def __init__(self, model_type="vit_l", checkpoint="models/sam_vit_l_0b3195.pth"):
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.model.to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(
                        model=self.model,
                        points_per_side=32,  # 减少采样点以加快速度
                        pred_iou_thresh=0.86,
                        stability_score_thresh=0.92,
                        crop_n_layers=1,
                        crop_n_points_downscale_factor=2,
                        min_mask_region_area=100,  # 过滤小掩码)
        )
    
    def get_obj_masks(self, rgb_raw: np.ndarray):
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
            masks = self.mask_generator.generate(rgb_np)
            print(f"Generated {len(masks)} masks")
            return masks
        
        except Exception as e:
            print(f"Error generating masks: {e}")
            return []

class DinoSamInterface:
    def __init__(
            self,
            dino_config_path = "models/GroundingDINO_SwinT_OGC.py",
            dino_checkpoint_path = "models/groundingdino_swint_ogc.pth",
            sam_model_type="vit_l",
            sam_checkpoint="models/sam_vit_l_0b3195.pth",
            device="cuda:0"
    ):
        self.device = device

        print(">---------- loading DINO model -------------<")
        self.dino_model = GroundingDINOModel(
                                    dino_config_path, 
                                    dino_checkpoint_path, 
                                    device=self.device
                            )
        print(">---------- DINO model loaded -------------<")
        print(">---------- loading SAM model -------------<")
        self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        self.sam_predictor = SamPredictor(self.sam_model)
        print(">---------- SAM model loaded -------------<")
    
    def preprocess_image(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if not isinstance(img, np.ndarray):
            raise TypeError(type(img))

        # CHW → HWC
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # RGBA → RGB
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        # float → uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        # RGB → BGR
        img = img[..., ::-1].copy()

        return img



    def dino_detect(self, 
                    rgb_image: np.ndarray, 
                    text_prompt: str
                ) -> Tuple[List[torch.Tensor], 
                            List[torch.Tensor]]:
        # print(
        # "[DINO INPUT DEBUG]",
        # "type:", type(rgb_image),
        # "dtype:", rgb_image.dtype,
        # "shape:", rgb_image.shape,
        # "ndim:", rgb_image.ndim,
        # "strides:", rgb_image.strides,
        # "contiguous:", rgb_image.flags['C_CONTIGUOUS'],
        # )
        rgb_image = np.ascontiguousarray(
            rgb_image.astype(np.uint8, copy=True)
        )
        # pil_image = Image.fromarray(rgb_image)
        detections = self.dino_model.predict_with_classes(
                                    rgb_image, 
                                    [text_prompt], 
                                    box_threshold=0.3, 
                                    text_threshold=0.25
                                )
        boxes = detections.xyxy # [N, 4] Tensor
        scores = detections.confidence # [N] Tensor
        return boxes, scores
    
    def sam_segment(self, 
                    rgb_image: np.ndarray, 
                    boxes: List[torch.Tensor], 
                    scores: List[torch.Tensor]
                ) -> Tuple[List[np.ndarray], 
                            List[np.ndarray], 
                            List[float]]:
        self.sam_predictor.set_image(rgb_image)
        masks = []
        valid_boxes = []
        vadid_scores = []    

        for i, box in enumerate(boxes):
            box_np = box
            
            mask, _, _ = self.sam_predictor.predict(
                                    point_coords=None,
                                    point_labels=None,
                                    box=box_np[None, :], # [1, 4]
                                    multimask_output=False
                                )
            
            masks.append(mask[0]) # [H, W]
            valid_boxes.append(box_np)
            vadid_scores.append(scores[i].item())

        return masks, valid_boxes, vadid_scores

    def detect_and_segment(self, 
                           rgb_image: np.ndarray, 
                           text_prompt: str, 
                           box_threshold: float = 0.3, 
                           text_threshold: float = 0.25,
                    ) -> Tuple[List[np.ndarray], 
                                List[np.ndarray], 
                                List[float]]:
        
        rgb_image = self.preprocess_image(rgb_image)

        boxes, scores = self.dino_detect(rgb_image, text_prompt)

        if len(boxes) == 0:
            return [], [], []
        
        masks, valid_boxes, valid_scores =  self.sam_segment(
                                                    rgb_image, 
                                                    boxes, 
                                                    scores
                                                )
        
        return masks, valid_boxes, valid_scores
        



