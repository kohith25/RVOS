#!/usr/bin/env python3
"""
tests/test_rewards.py - Rigorous Reward Function Unit Tests

Verifies P4 Requirements:
(a) stop-only < any episode calling sam2
(b) Reward monotonicity in final-mask J (IoU)
(c) Efficiency: correct short episode beats correct long episode
(d) Null-sample behavior: rewards empty predictions, penalizes false alarms
"""

import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.rewards import (
    compute_episodic_reward,
    compute_format_reward,
    compute_segmentation_reward,
    compute_mask_iou,
    RewardWeights
)


def create_synthetic_mask(shape=(10, 224, 224), box=(50, 50, 150, 150)):
    mask = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = box
    mask[:, y1:y2, x1:x2] = 1
    return mask


def test_bare_stop_vs_sam2_episode():
    """(a) Test: Bare 'stop' receives severe penalty and scores less than any valid sam2 episode."""
    gt_masks = create_synthetic_mask()
    expression = "the person singing in the center"

    # Bare stop trajectory: only 1 turn, 'stop'
    bare_stop_turns = [{"action": "stop", "args": {}, "is_valid": True}]
    res_bare_stop = compute_episodic_reward(
        turns=bare_stop_turns,
        final_pred_masks=None,
        gt_masks=gt_masks,
        expression=expression,
        is_null=False
    )

    # Valid episode calling referdino, sam2, then stop
    valid_sam2_turns = [
        {"action": "call_referdino", "args": {"text": expression}, "is_valid": True},
        {"action": "call_sam2", "args": {"candidate_id": 0}, "is_valid": True},
        {"action": "stop", "args": {}, "is_valid": True}
    ]
    # Prediction with modest IoU (~0.5)
    pred_masks = create_synthetic_mask(box=(60, 60, 160, 160))
    res_valid = compute_episodic_reward(
        turns=valid_sam2_turns,
        final_pred_masks=pred_masks,
        gt_masks=gt_masks,
        expression=expression,
        is_null=False
    )

    assert res_bare_stop["R_f"] <= -1.0, f"Bare stop format reward should be <= -1.0, got {res_bare_stop['R_f']}"
    assert res_bare_stop["reward"] < res_valid["reward"], (
        f"Bare stop reward ({res_bare_stop['reward']}) must be strictly less than valid sam2 ({res_valid['reward']})"
    )


def test_reward_monotonicity_in_mask_iou():
    """(b) Test: Reward is strictly monotonic with respect to mask IoU (J)."""
    gt_masks = create_synthetic_mask(box=(40, 40, 160, 160))
    expression = "guitar on the left"

    turns = [
        {"action": "call_referdino", "args": {"text": expression}, "is_valid": True},
        {"action": "call_sam2", "args": {"candidate_id": 0}, "is_valid": True},
        {"action": "stop", "args": {}, "is_valid": True}
    ]

    # Vary predictions with progressively higher IoU
    boxes = [
        (10, 10, 50, 50),     # low IoU
        (30, 30, 130, 130),   # medium-low
        (35, 35, 155, 155),   # medium-high
        (40, 40, 160, 160),   # perfect IoU (1.0)
    ]

    rewards = []
    ious = []
    for b in boxes:
        pred_m = create_synthetic_mask(box=b)
        res = compute_episodic_reward(
            turns=turns,
            final_pred_masks=pred_m,
            gt_masks=gt_masks,
            expression=expression,
            is_null=False
        )
        ious.append(res["mask_iou"])
        rewards.append(res["reward"])

    # Assert IoUs are increasing
    assert all(ious[i] < ious[i + 1] for i in range(len(ious) - 1))
    # Assert Rewards are strictly increasing
    assert all(rewards[i] < rewards[i + 1] for i in range(len(rewards) - 1)), (
        f"Rewards are not strictly monotonic! IoUs: {ious}, Rewards: {rewards}"
    )


def test_efficiency_short_vs_long_episode():
    """(c) Test: Correct short episode beats correct long redundant episode (due to tool costs)."""
    gt_masks = create_synthetic_mask(box=(50, 50, 150, 150))
    pred_masks = create_synthetic_mask(box=(50, 50, 150, 150))  # Same high quality
    expression = "man singing"

    # Short episode: 3 turns
    short_turns = [
        {"action": "call_referdino", "args": {"text": expression}, "is_valid": True},
        {"action": "call_sam2", "args": {"candidate_id": 0}, "is_valid": True},
        {"action": "stop", "args": {}, "is_valid": True}
    ]
    res_short = compute_episodic_reward(
        turns=short_turns,
        final_pred_masks=pred_masks,
        gt_masks=gt_masks,
        expression=expression,
        is_null=False
    )

    # Long episode: 7 turns (redundant inspects and refinements)
    long_turns = [
        {"action": "inspect_frames", "args": {"start": 0, "end": 4}, "is_valid": True},
        {"action": "inspect_frames", "args": {"start": 5, "end": 9}, "is_valid": True},
        {"action": "refine_description", "args": {"text": "man"}, "is_valid": True},
        {"action": "refine_description", "args": {"text": "man singing"}, "is_valid": True},
        {"action": "call_referdino", "args": {"text": expression}, "is_valid": True},
        {"action": "call_sam2", "args": {"candidate_id": 0}, "is_valid": True},
        {"action": "stop", "args": {}, "is_valid": True}
    ]
    res_long = compute_episodic_reward(
        turns=long_turns,
        final_pred_masks=pred_masks,
        gt_masks=gt_masks,
        expression=expression,
        is_null=False
    )

    assert res_short["C_tool"] < res_long["C_tool"]
    assert res_short["reward"] > res_long["reward"], (
        f"Short episode reward ({res_short['reward']}) must beat long episode ({res_long['reward']})"
    )


def test_null_sample_handling():
    """(d) Test: Null-sample behavior correctly rewards empty prediction and penalizes false alarm."""
    null_gt_masks = np.zeros((10, 224, 224), dtype=bool)
    expression = "null"

    turns = [
        {"action": "call_referdino", "args": {"text": expression}, "is_valid": True},
        {"action": "call_sam2", "args": {"candidate_id": 0}, "is_valid": True},
        {"action": "stop", "args": {}, "is_valid": True}
    ]

    # Correct empty prediction
    empty_pred = np.zeros((10, 224, 224), dtype=np.uint8)
    res_empty = compute_episodic_reward(
        turns=turns,
        final_pred_masks=empty_pred,
        gt_masks=null_gt_masks,
        expression=expression,
        is_null=True
    )

    # False alarm prediction
    false_alarm_pred = create_synthetic_mask(box=(30, 30, 180, 180))
    res_false_alarm = compute_episodic_reward(
        turns=turns,
        final_pred_masks=false_alarm_pred,
        gt_masks=null_gt_masks,
        expression=expression,
        is_null=True
    )

    assert res_empty["R_m"] > 0.0, f"Expected positive R_m for empty prediction on null video, got {res_empty['R_m']}"
    assert res_false_alarm["R_m"] < 0.0, f"Expected negative R_m for false alarm on null video, got {res_false_alarm['R_m']}"
    assert res_empty["reward"] > res_false_alarm["reward"]


def test_evidence_reward_gt_overlap():
    """FIX 7(b): Test R_e scores positively when query matches GT object location, negatively when it doesn't."""
    # GT object is on the left (x: 10 to 60 in 224 width -> center ~ 0.15)
    gt_masks_left = create_synthetic_mask(box=(10, 50, 60, 150))
    turns_left_match = [{"action": "retrieve_evidence", "args": {"query": "guitar on the left"}, "is_valid": True}]
    r_left = compute_episodic_reward(turns=turns_left_match, final_pred_masks=gt_masks_left, gt_masks=gt_masks_left, expression="guitar on the left", is_null=False)

    turns_wrong_match = [{"action": "retrieve_evidence", "args": {"query": "drum on the right"}, "is_valid": True}]
    r_wrong = compute_episodic_reward(turns=turns_wrong_match, final_pred_masks=gt_masks_left, gt_masks=gt_masks_left, expression="guitar on the left", is_null=False)

    assert r_left["R_e"] > r_wrong["R_e"]
    assert r_left["R_e"] > 0.0
    assert r_wrong["R_e"] < 0.0


def test_dataset_null_detection_no_string_hack():
    """FIX 7(e): Ref expression containing word 'nullify' must NOT be falsely marked as a null audio sample."""
    from rl_grpo.dataset_adapter import TGSRLDataset
    base_samples = [
        {"uid": "vid_001_fid_0", "ref": "the sound does not nullify the trumpet", "instruction": "play", "split": "test_s"},
        {"uid": "null_vid_002_fid_0", "ref": "guitar playing", "instruction": "play", "split": "test_n"}
    ]
    ds = TGSRLDataset(base_samples, use_mock=True)

    item0 = ds[0]
    assert item0["is_null"] is False, "A sentence with 'nullify' was falsely marked as null!"

    item1 = ds[1]
    assert item1["is_null"] is True, "Sample with uid starting with null_ must be marked as null!"


def test_dataset_missing_multimodal_raises_without_mock():
    """FIX 7(f): When media files are missing and use_mock=False, dataset must RAISE FileNotFoundError."""
    from rl_grpo.dataset_adapter import TGSRLDataset
    base_samples = [
        {"uid": "vid_missing_media", "ref": "dog", "image_path_list": ["./nonexistent_frame.jpg"]}
    ]
    ds = TGSRLDataset(base_samples, use_mock=False)
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
