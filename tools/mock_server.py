#!/usr/bin/env python3
"""
tools/mock_server.py - In-Memory Mock Tool Server

Provides an in-process, zero-dependency implementation of all tools:
- ReferDINO candidate grounding
- SAM2 mask segmentation
- KG multimodal subgraph retrieval

Used for testing, CPU execution, and standalone fallback when the separate
dino tool server daemon is not running.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from tools.referdino_tool import get_referdino_tool, Candidate
from tools.sam2_tool import get_sam2_tool
from kg.retriever import retrieve


class MockToolServer:
    def __init__(self, media_dir="./REFAVS/media", kg_dir="kg_cache"):
        self.media_dir = media_dir
        self.kg_dir = kg_dir
        self.referdino = get_referdino_tool(use_mock=True)
        self.sam2 = get_sam2_tool(use_mock=True, media_dir=media_dir)

    def referdino_ground(self, video_id: str, text: str) -> List[Dict[str, Any]]:
        candidates = self.referdino.run(video_id, text)
        return [c.to_dict() for c in candidates]

    def sam2_segment(self, video_id: str, candidate_boxes: List[List[float]], keyframe: Optional[int] = None) -> List[List[List[int]]]:
        masks_arr = self.sam2.refine(video_id, candidate_boxes, keyframe=keyframe)  # (10, 224, 224) uint8
        return masks_arr.tolist()

    def kg_retrieve(self, video_id: str, query: str, k_hops: int = 1) -> Dict[str, Any]:
        return retrieve(video_id, query, k_hops=k_hops, kg_dir=self.kg_dir)
