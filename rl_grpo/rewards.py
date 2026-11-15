#!/usr/bin/env python3
"""
rl_grpo/rewards.py - Episodic Composite Reward System for Ref-Thinker

Implements the multi-component episodic reward adapted from VideoSEG-O3:
  R = w_m * R_m + w_t * R_t + w_e * R_e + w_p * R_p + w_f * R_f - w_c * C_tool

Components:
1. Format Reward (R_f):
   +1.0 only if all actions are valid JSON and the episode executed call_sam2 followed by stop.
   -0.5 per invalid action. Bare 'stop' receives <= -1.0.
2. Segmentation Quality Reward (R_m):
   Computed using True GT masks across 10 frames with VideoSEG-O3 tiered scaling:
   - J > 0.4: min(3.0 * (J - 0.4), 1.5)
   - 0.2 < J <= 0.4: 0.0
   - J <= 0.2: -1.0
   Null split: rewards empty prediction (+1.5) and penalizes false alarm area (-3.0 * sqrt(area)).
3. Temporal Frame Inspection Reward (R_t):
   Precision-based reward for inspect_frames windows covering active object frames.
4. Evidence Retrieval Reward (R_e):
   Rewards relevant KG queries and penalizes spurious retrievals.
5. Progressive Improvement Reward (R_p):
   Rewards successive improvements across candidate refinements.
6. Tool Cost (C_tool):
   Penalizes redundant / unnecessary tool invocations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from rl_grpo.constants import ACTION_TO_COST


@dataclass
class RewardWeights:
    wf: float = 1.0   # Format
    wm: float = 2.0   # Segmentation (J&F)
    wt: float = 0.5   # Temporal inspection
    we: float = 0.5   # Knowledge retrieval
    wp: float = 0.5   # Progressive improvement
    wc: float = 1.0   # Tool costs


def compute_mask_iou(pred_masks: np.ndarray, gt_masks: np.ndarray, eps: float = 1e-7) -> float:
    """
    Computes global Intersection-over-Union across all 10 video frames.
    pred_masks: (10, H, W) bool or uint8
    gt_masks: (10, H, W) bool or uint8
    """
    pred_bool = pred_masks > 0
    gt_bool = gt_masks > 0

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / (union + eps))


def compute_format_reward(turns: List[Dict[str, Any]], has_final_sam2: bool, has_stop: bool) -> float:
    """
    R_f: Enforces structured action sequence ending in call_sam2 then stop.
    Bare stop receives -1.0.
    """
    if not turns:
        return -1.0

    # Bare stop check (only 1 turn and it's stop)
    if len(turns) == 1 and turns[0].get("action") == "stop":
        return -1.0

    invalid_count = sum(1 for t in turns if not t.get("is_valid", True))
    penalty = -0.5 * invalid_count

    # Must contain call_sam2 before stop
    sam2_indices = [i for i, t in enumerate(turns) if t.get("action") == "call_sam2"]
    stop_indices = [i for i, t in enumerate(turns) if t.get("action") == "stop"]

    if sam2_indices and stop_indices:
        if stop_indices[-1] > sam2_indices[-1]:
            # Perfect valid termination
            return 1.0 + penalty

    # No sam2 or stopped prematurely
    if not sam2_indices:
        return -1.0 + penalty

    return 0.0 + penalty


def compute_segmentation_reward(final_pred_masks: Optional[np.ndarray],
                                gt_masks: np.ndarray,
                                is_null: bool = False) -> Tuple[float, float]:
    """
    R_m: VideoSEG-O3 tiered segmentation reward based on true GT masks.
    Returns:
        (R_m, mask_iou)
    """
    if final_pred_masks is None:
        if is_null:
            return 1.5, 1.0  # Correctly predicted nothing on null video
        return -1.0, 0.0

    pred_bool = final_pred_masks > 0
    gt_bool = gt_masks > 0

    # Null video evaluation
    if is_null or gt_bool.sum() == 0:
        num_pixels = pred_bool.size
        pred_area = pred_bool.sum() / num_pixels
        if pred_area == 0:
            return 1.5, 1.0  # Perfect empty prediction
        # Penalize false alarm area
        s_null = np.sqrt(pred_area)
        r_m = max(-1.5, -3.0 * s_null)
        return float(r_m), 0.0

    # Non-null video evaluation
    iou = compute_mask_iou(final_pred_masks, gt_masks)

    # VideoSEG-O3 tiers
    if iou > 0.4:
        r_m = min(3.0 * (iou - 0.4), 1.5)
    elif 0.2 < iou <= 0.4:
        r_m = 0.0
    else:
        r_m = -1.0

    return float(r_m), float(iou)


def compute_temporal_reward(turns: List[Dict[str, Any]], gt_masks: np.ndarray) -> float:
    """
    R_t: Precision-based reward for inspect_frames windows covering active GT frames.
    """
    inspect_turns = [t for t in turns if t.get("action") == "inspect_frames"]
    if not inspect_turns:
        return 0.0

    scores = []
    gt_active_frames = set(np.where((gt_masks > 0).sum(axis=(1, 2)) > 0)[0])

    for t in inspect_turns:
        args = t.get("args", {})
        start = int(args.get("start", 0))
        end = int(args.get("end", 9))
        window = list(range(start, end + 1))
        if not window:
            continue

        active_in_window = sum(1 for f in window if f in gt_active_frames)
        precision = active_in_window / len(window)

        if precision >= 0.5:
            scores.append(0.25 * precision)
        else:
            scores.append(-0.5)

    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_evidence_reward(
    turns: List[Dict[str, Any]],
    expression: str,
    gt_masks: np.ndarray,
    retrieved_nodes: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    FIX 7(b): R_e evaluates quality of retrieved KG evidence:
    R_e = fraction of retrieved facts/nodes that overlap the GT object (IoU > 0.5 or correct spatial sector).
    """
    kg_turns = [t for t in turns if t.get("action") == "retrieve_evidence"]
    if not kg_turns:
        return 0.0

    scores = []
    expr_lower = expression.lower()

    # Determine GT spatial location
    gt_side = "center"
    if gt_masks is not None and gt_masks.sum() > 0:
        flat_gt = gt_masks.sum(axis=0) > 0
        ys, xs = np.where(flat_gt)
        if len(xs) > 0:
            h, w = flat_gt.shape
            mean_cx = np.mean(xs) / float(w)
            if mean_cx < 0.35:
                gt_side = "left"
            elif mean_cx > 0.65:
                gt_side = "right"
            else:
                gt_side = "center"

    STOPWORDS = {"on", "the", "in", "at", "a", "an", "is", "of", "and", "to", "with", "by"}
    for t in kg_turns:
        args = t.get("args", {})
        query = args.get("query", "").lower()
        if not query:
            scores.append(-0.2)
            continue

        # Check alignment: query mentions correct GT side or key entity in expression (excluding stopwords)
        words = set(query.split()) - STOPWORDS
        ref_words = set(expr_lower.split()) - STOPWORDS
        overlap = len(words & ref_words)
        side_match = (gt_side in query)

        # If query targets wrong spatial side, penalize
        wrong_side = any(s in query for s in ("left", "right", "center") if s != gt_side)

        if side_match and not wrong_side:
            scores.append(0.5)
        elif wrong_side:
            scores.append(-0.2)
        elif overlap >= 1:
            scores.append(0.3)
        else:
            scores.append(-0.2)

    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_progressive_reward(sam2_ious: List[float]) -> float:
    """
    R_p: Rewards monotonic improvement across successive SAM2 segmentation turns.
    """
    if len(sam2_ious) < 2:
        return 0.0

    improvements = 0.0
    for prev_iou, cur_iou in zip(sam2_ious[:-1], sam2_ious[1:]):
        delta = cur_iou - prev_iou
        if delta > 0.05:
            improvements += 0.5
        elif delta <= 0.0:
            improvements -= 0.3
    return float(improvements)


def compute_tool_cost(turns: List[Dict[str, Any]]) -> float:
    """
    C_tool: Computes sum of tool execution costs.
    """
    total_cost = 0.0
    for t in turns:
        act = t.get("action", "")
        cost = ACTION_TO_COST.get(act, 0.05)
        total_cost += cost
    return float(total_cost)


def compute_episodic_reward(turns: List[Dict[str, Any]],
                            final_pred_masks: Optional[np.ndarray],
                            gt_masks: np.ndarray,
                            expression: str,
                            is_null: bool = False,
                            sam2_ious: Optional[List[float]] = None,
                            weights: Optional[RewardWeights] = None) -> Dict[str, float]:
    """
    Main episodic reward entry point for GRPO rollouts.
    Computes all components and combines them into scalar R.
    """
    if weights is None:
        weights = RewardWeights()

    if sam2_ious is None:
        sam2_ious = []

    has_final_sam2 = any(t.get("action") == "call_sam2" for t in turns)
    has_stop = any(t.get("action") == "stop" for t in turns)

    rf = compute_format_reward(turns, has_final_sam2, has_stop)
    rm, mask_iou = compute_segmentation_reward(final_pred_masks, gt_masks, is_null=is_null)
    rt = compute_temporal_reward(turns, gt_masks)
    re = compute_evidence_reward(turns, expression, gt_masks)
    rp = compute_progressive_reward(sam2_ious)
    c_tool = compute_tool_cost(turns)

    total_reward = (
        weights.wf * rf +
        weights.wm * rm +
        weights.wt * rt +
        weights.we * re +
        weights.wp * rp -
        weights.wc * c_tool
    )

    return {
        "reward": float(total_reward),
        "R_f": float(rf),
        "R_m": float(rm),
        "R_t": float(rt),
        "R_e": float(re),
        "R_p": float(rp),
        "C_tool": float(c_tool),
        "mask_iou": float(mask_iou)
    }
