from __future__ import annotations
#!/usr/bin/env python3
"""
tools/mock_tools.py - Dedicated Mock Implementations for Testing & CPU Execution

Contains mock candidate generators, mock mask rasterizers, and mock tool servers.
These are invoked ONLY when --mock is passed explicitly.
"""

import os
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from kg.retriever import retrieve


@dataclass
class MockCandidate:
    cand_id: int
    boxes: List[List[float]]
    masks: Optional[List[Any]]
    score: float
    mean_area: float
    side: str

    def to_dict(self):
        return asdict(self)


class MockReferDINOTool:
    """Mock ReferDINO tool producing deterministic synthetic candidate tubelets."""
    def __init__(self):
        pass

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

    def run(self, video_id: str, text: str) -> List[MockCandidate]:
        text_lower = text.lower()
        query_hash = int(hashlib.md5((video_id + text_lower).encode("utf-8")).hexdigest(), 16)

        # Candidate 0: Center dominant object
        c0_boxes = []
        for t in range(10):
            jitter_x = 0.01 * ((query_hash + t) % 5 - 2)
            jitter_y = 0.01 * ((query_hash + t * 2) % 5 - 2)
            c0_boxes.append([
                round(max(0.0, 0.28 + jitter_x), 3),
                round(max(0.0, 0.18 + jitter_y), 3),
                round(min(1.0, 0.72 + jitter_x), 3),
                round(min(1.0, 0.82 + jitter_y), 3)
            ])
        score_c0 = 0.75 if any(w in text_lower for w in ["center", "middle", "man", "woman", "person"]) else 0.65

        # Candidate 1: Left object
        c1_boxes = []
        for t in range(10):
            jitter_x = 0.01 * ((query_hash + t * 3) % 4 - 2)
            c1_boxes.append([
                round(max(0.0, 0.08 + jitter_x), 3),
                round(max(0.0, 0.22), 3),
                round(min(1.0, 0.42 + jitter_x), 3),
                round(min(1.0, 0.78), 3)
            ])
        score_c1 = 0.82 if any(w in text_lower for w in ["left", "guitar", "instrument", "cat"]) else 0.52

        # Candidate 2: Right object
        c2_boxes = []
        for t in range(10):
            jitter_x = 0.01 * ((query_hash + t * 4) % 4 - 2)
            c2_boxes.append([
                round(max(0.0, 0.58 + jitter_x), 3),
                round(max(0.0, 0.25), 3),
                round(min(1.0, 0.92 + jitter_x), 3),
                round(min(1.0, 0.75), 3)
            ])
        score_c2 = 0.78 if any(w in text_lower for w in ["right", "second", "dog"]) else 0.48

        candidates_raw = [
            (c0_boxes, score_c0),
            (c1_boxes, score_c1),
            (c2_boxes, score_c2)
        ]
        candidates_raw.sort(key=lambda x: x[1], reverse=True)

        candidates = []
        for idx, (boxes, score) in enumerate(candidates_raw):
            side = self._compute_side(boxes)
            mean_area = self._compute_mean_area(boxes)
            candidates.append(MockCandidate(
                cand_id=idx,
                boxes=boxes,
                masks=None,
                score=round(score, 3),
                mean_area=round(mean_area, 3),
                side=side
            ))
        return candidates


class MockSAM2Tool:
    """Mock SAM2 tool rasterizing bounding boxes into binary masks for tests."""
    def __init__(self, media_dir="./REFAVS/media"):
        self.media_dir = media_dir

    def rasterize_box_to_mask(self, box, shape=(224, 224)):
        h, w = shape
        mask = np.zeros(shape, dtype=np.uint8)
        if len(box) != 4:
            return mask

        x1 = int(box[0] * w)
        y1 = int(box[1] * h)
        x2 = int(box[2] * w)
        y2 = int(box[3] * h)

        if x2 <= x1 or y2 <= y1:
            return mask

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = max(1, (x2 - x1) // 2)
        ry = max(1, (y2 - y1) // 2)

        if cv2 is not None:
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        else:
            mask[y1:y2, x1:x2] = 1
        return mask

    def refine(self, video_id: str, candidate_boxes: List[List[float]], keyframe: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        masks = []
        logits = []
        for t in range(10):
            box = candidate_boxes[t] if t < len(candidate_boxes) else [0, 0, 0, 0]
            m = self.rasterize_box_to_mask(box, shape=(224, 224))
            masks.append(m)
            # Sigmoid-equivalent logits: 0.9 inside mask, 0.1 outside
            l = np.where(m > 0, 0.9, 0.1).astype(np.float32)
            logits.append(l)
        return np.stack(masks, axis=0), np.stack(logits, axis=0)


class MockToolServer:
    """Mock server implementing the full tool suite in-process for tests."""
    def __init__(self, media_dir="./REFAVS/media", kg_dir="kg_cache"):
        self.media_dir = media_dir
        self.kg_dir = kg_dir
        self.referdino = MockReferDINOTool()
        self.sam2 = MockSAM2Tool(media_dir=media_dir)

    def referdino_ground(self, video_id: str, text: str) -> List[Dict[str, Any]]:
        candidates = self.referdino.run(video_id, text)
        return [c.to_dict() for c in candidates]

    def sam2_segment(self, video_id: str, candidate_boxes: List[List[float]], keyframe: Optional[int] = None) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
        masks_arr, logits_arr = self.sam2.refine(video_id, candidate_boxes, keyframe=keyframe)
        return masks_arr.tolist(), logits_arr.tolist()

    def kg_retrieve(self, video_id: str, query: str, k_hops: int = 1) -> Dict[str, Any]:
        return retrieve(video_id, query, k_hops=k_hops, kg_dir=self.kg_dir)
