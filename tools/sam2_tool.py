#!/usr/bin/env python3
"""
tools/sam2_tool.py - SAM2 Video Object Segmentation Tool

Refines candidate bounding box sequences into pixel-level binary masks across 10 video frames.
Uses SAM2ImagePredictor in the dino environment when use_mock=False.
Delegates to MockSAM2Tool ONLY when use_mock=True is explicitly passed.
Output masks match the original image's native resolution (10, H, W).
"""

import os
from typing import List, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
except ImportError:
    cv2 = None

from tools.mock_tools import MockSAM2Tool


class SAM2Tool:
    def __init__(
        self,
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        device="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
        use_mock=False,
        media_dir="./REFAVS/media",
    ):
        self.sam2_config = sam2_config
        self.sam2_checkpoint = sam2_checkpoint
        self.device = device
        self.use_mock = use_mock
        self.media_dir = media_dir
        self.predictor = None

        if self.use_mock:
            self._mock_impl = MockSAM2Tool(media_dir=media_dir)
        else:
            self._mock_impl = None
            self._load_sam2()

    def _load_sam2(self):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for real SAM2 segmentation.")
        if not os.path.exists(self.sam2_checkpoint):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at: {self.sam2_checkpoint}. "
                "Ensure weights are downloaded or pass --mock to run with mocks."
            )

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def refine(
        self,
        video_id: str,
        candidate_boxes: List[List[float]],
        keyframe: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates 10 binary masks of shape (10, H, W) matching native video resolution.
        """
        if self.use_mock:
            return self._mock_impl.refine(video_id, candidate_boxes, keyframe=keyframe)

        if self.predictor is None:
            raise RuntimeError("SAM2 predictor is not loaded.")

        vid = video_id.rsplit("_", 2)[0] if "_" in video_id else video_id
        frame_dir = os.path.join(self.media_dir, vid, "frames")

        if not os.path.exists(frame_dir):
            raise FileNotFoundError(f"Video frames directory not found: {frame_dir}")

        # Check native image shape from frame 0
        first_frame = os.path.join(frame_dir, "0.jpg")
        if os.path.exists(first_frame) and cv2 is not None:
            sample_img = cv2.imread(first_frame)
            native_h, native_w = sample_img.shape[:2]
        else:
            native_h, native_w = 224, 224

        pred_masks = []
        for img_id in range(10):
            img_path = os.path.join(frame_dir, f"{img_id}.jpg")
            box_norm = candidate_boxes[img_id] if img_id < len(candidate_boxes) else [0, 0, 0, 0]

            if not os.path.exists(img_path) or box_norm == [0, 0, 0, 0]:
                pred_masks.append(np.zeros((native_h, native_w), dtype=np.uint8))
                continue

            image_bgr = cv2.imread(img_path) if cv2 is not None else None
            if image_bgr is None:
                pred_masks.append(np.zeros((native_h, native_w), dtype=np.uint8))
                continue

            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Convert normalized box to pixel coordinates [x1, y1, x2, y2]
            box_pixel = np.array([[
                box_norm[0] * w,
                box_norm[1] * h,
                box_norm[2] * w,
                box_norm[3] * h,
            ]], dtype=np.float32)

            self.predictor.set_image(image_rgb)
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_pixel,
                multimask_output=False,
            )

            if masks is None or len(masks) == 0:
                pred_masks.append(np.zeros((h, w), dtype=np.uint8))
            else:
                pred_masks.append(masks[0].astype(np.uint8))

        return np.stack(pred_masks, axis=0)


_default_sam2_tool = None


def get_sam2_tool(use_mock=False, media_dir="./REFAVS/media"):
    global _default_sam2_tool
    if _default_sam2_tool is None:
        _default_sam2_tool = SAM2Tool(use_mock=use_mock, media_dir=media_dir)
    return _default_sam2_tool
