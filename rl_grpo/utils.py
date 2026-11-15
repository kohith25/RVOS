from __future__ import annotations
import math
import re
from typing import List, Optional


def parse_gt_bbox(text: str) -> Optional[List[float]]:
    if not text:
        return None

    # Match tuple-like "(x1,y1),(x2,y2)"
    m = re.search(r"\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)\s*,\s*\(([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\)", text)
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]

    # Match list-like "[x1, y1, x2, y2]"
    m = re.search(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", text)
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]

    return None


def box_iou_xyxy(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / union


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
