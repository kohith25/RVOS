#!/usr/bin/env python3
"""
tools/referdino_tool.py - ReferDINO / GroundingDINO Grounding Tool

Extracts spatio-temporal candidate tubelets across 10 video frames given a text expression.
Uses real GroundingDINO from the dino environment when use_mock=False.
Delegates to MockReferDINOTool ONLY when use_mock=True is explicitly passed.
"""

import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import numpy as np

try:
    import torch
    from torchvision.ops import box_convert
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from tools.mock_tools import MockCandidate, MockReferDINOTool


@dataclass
class Candidate:
    cand_id: int
    boxes: List[List[float]]
    masks: Optional[List[Any]]
    score: float
    mean_area: float
    side: str

    def to_dict(self):
        return asdict(self)


class ReferDINOTool:
    def __init__(
        self,
        config_path="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path="gdino_checkpoints/groundingdino_swint_ogc.pth",
        device="cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu",
        media_dir="./REFAVS/media",
        box_threshold=0.15,
        text_threshold=0.25,
        use_mock=False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.media_dir = media_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_mock = use_mock
        self.model = None

        if self.use_mock:
            self._mock_impl = MockReferDINOTool()
        else:
            self._mock_impl = None
            self._load_model()

    def _load_model(self):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for real ReferDINO/GroundingDINO inference.")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found at: {self.checkpoint_path}. "
                "Ensure weights are placed in gdino_checkpoints/ or pass --mock to run with mocks."
            )

        from grounding_dino.groundingdino.util.inference import load_model
        self.model = load_model(
            model_config_path=self.config_path,
            model_checkpoint_path=self.checkpoint_path,
            device=self.device
        )

    def _compute_side(self, boxes):
        center_xs = [((b[0] + b[2]) / 2.0) for b in boxes if len(b) == 4]
        if not center_xs:
            return "center"
        mean_cx = sum(center_xs) / len(center_xs)
        if mean_cx < 0.35:
            return "left"
        elif mean_cx > 0.65:
            return "right"
        else:
            return "center"

    def _compute_mean_area(self, boxes):
        areas = [max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])) for b in boxes if len(b) == 4]
        return sum(areas) / len(areas) if areas else 0.0

    def run(self, video_id: str, text: str) -> List[Candidate]:
        if self.use_mock:
            mock_cands = self._mock_impl.run(video_id, text)
            return [
                Candidate(
                    cand_id=c.cand_id,
                    boxes=c.boxes,
                    masks=c.masks,
                    score=c.score,
                    mean_area=c.mean_area,
                    side=c.side,
                )
                for c in mock_cands
            ]

        # Real GroundingDINO Inference Path
        if self.model is None:
            raise RuntimeError("GroundingDINO model is not loaded.")

        from grounding_dino.groundingdino.util.inference import load_image, predict
        vid = video_id.rsplit("_", 2)[0] if "_" in video_id else video_id
        frame_dir = os.path.join(self.media_dir, vid, "frames")

        if not os.path.exists(frame_dir):
            raise FileNotFoundError(f"Video frames directory not found: {frame_dir}")

        frame_detections = []
        for img_id in range(10):
            img_path = os.path.join(frame_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                frame_detections.append([])
                continue

            _, image = load_image(img_path)
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

            # Convert normalized cxcywh to xyxy
            if boxes.shape[0] > 0:
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
                scores = logits.cpu().numpy()
                frame_dets = [{"box": xyxy[i].tolist(), "score": float(scores[i])} for i in range(len(scores))]
            else:
                frame_dets = []
            frame_detections.append(frame_dets)

        # Track top candidate tubelets across 10 frames
        candidates = []
        for cand_idx in range(3):
            cand_boxes = []
            cand_scores = []
            for t in range(10):
                dets = frame_detections[t]
                if len(dets) > cand_idx:
                    cand_boxes.append(dets[cand_idx]["box"])
                    cand_scores.append(dets[cand_idx]["score"])
                elif len(dets) > 0:
                    cand_boxes.append(dets[0]["box"])
                    cand_scores.append(dets[0]["score"])
                else:
                    cand_boxes.append([0.0, 0.0, 0.0, 0.0])
                    cand_scores.append(0.0)

            mean_score = sum(cand_scores) / len(cand_scores) if cand_scores else 0.0
            side = self._compute_side(cand_boxes)
            mean_area = self._compute_mean_area(cand_boxes)

            candidates.append(Candidate(
                cand_id=cand_idx,
                boxes=cand_boxes,
                masks=None,
                score=round(mean_score, 3),
                mean_area=round(mean_area, 3),
                side=side,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        # Re-index cand_id by sorted rank
        for i, c in enumerate(candidates):
            c.cand_id = i

        return candidates


_default_referdino = None


def get_referdino_tool(use_mock=False):
    global _default_referdino
    if _default_referdino is None:
        _default_referdino = ReferDINOTool(use_mock=use_mock)
    return _default_referdino
