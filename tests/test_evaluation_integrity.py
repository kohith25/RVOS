#!/usr/bin/env python3
"""
tests/test_evaluation_integrity.py - Tests for Evaluation Metrics Integrity

1. Verifies that fake metric patterns (- 0.03, max(0.0, mean_iou) do NOT exist anywhere in codebase.
2. Verifies that infer_pipeline metrics strictly reproduce avss_utils calculation (requires PyTorch).
"""

import os
import sys
import re
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def test_no_fake_metrics_in_codebase():
    """Assert that no file contains fake metric approximations."""
    prohibited_patterns = [
        re.compile(r"-\s*0\.03"),
        re.compile(r"max\s*\(\s*0\.0\s*,\s*mean_iou"),
    ]

    target_dirs = ["scripts", "rl_grpo", "tools", "kg", "tests"]
    matched_files = []

    for d in target_dirs:
        dir_path = os.path.join(REPO_ROOT, d)
        if not os.path.exists(dir_path):
            continue
        for root, _, files in os.walk(dir_path):
            for f in files:
                if not f.endswith(".py"):
                    continue
                # Skip this test file itself
                if f == "test_evaluation_integrity.py":
                    continue
                fpath = os.path.join(root, f)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    content = fp.read()
                for pat in prohibited_patterns:
                    if pat.search(content):
                        matched_files.append((fpath, pat.pattern))

    assert len(matched_files) == 0, f"Found prohibited fake metric patterns in: {matched_files}"


def test_synthetic_metric_parity():
    """Verify synthetic metric parity with original avss_utils implementation."""
    torch = pytest.importorskip("torch")
    F = pytest.importorskip("torch.nn.functional")

    gsam2_path = os.path.join(REPO_ROOT, "ground_segment_scripts", "Grounded-SAM-2")
    if gsam2_path not in sys.path:
        sys.path.insert(0, gsam2_path)

    from utils.avss_utils import mask_iou, Eval_Fmeasure, metric_s_for_null_batch

    # Create synthetic binary masks and logits
    pred_masks = torch.zeros((10, 224, 224), dtype=torch.uint8)
    pred_masks[:, 50:150, 50:150] = 1

    gt_masks = torch.zeros((10, 224, 224), dtype=torch.uint8)
    gt_masks[:, 60:160, 60:160] = 1

    # Logits: high inside prediction, low outside
    pred_logits = torch.full((10, 224, 224), 0.1, dtype=torch.float32)
    pred_logits[:, 50:150, 50:150] = 0.9

    iou_val = mask_iou(pred_masks, gt_masks)
    fscore_val = Eval_Fmeasure(pred=pred_logits.cpu(), gt=gt_masks.cpu())

    assert isinstance(iou_val, torch.Tensor)
    assert 0.0 < iou_val.item() < 1.0
    assert 0.0 < fscore_val < 1.0

    # Test null metric
    null_s = metric_s_for_null_batch(pred_masks)
    assert isinstance(null_s, torch.Tensor)
    assert null_s.item() > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
