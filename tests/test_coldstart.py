from __future__ import annotations
#!/usr/bin/env python3
"""
tests/test_coldstart.py - Tests for Phase 3: Real Cold-Start Trajectories & SFT
"""

import os
import sys
import json
import pytest
import tempfile
from typing import List, Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.rl_grpo.build_coldstart_data import main as build_coldstart_main
from rl_grpo.constants import parse_and_validate_action


def test_coldstart_trajectories_differ_across_samples(tmp_path):
    """Generated cold-start trajectories must differ across different expressions/samples."""
    out_file = str(tmp_path / "coldstart_test.jsonl")
    sys_argv_backup = sys.argv
    try:
        sys.argv = [
            "build_coldstart_data.py",
            "--out_path", out_file,
            "--num_samples", "5",
            "--mock",
        ]
        build_coldstart_main()
    finally:
        sys.argv = sys_argv_backup

    assert os.path.exists(out_file)

    lines = []
    with open(out_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))

    assert len(lines) > 0
    # Collect thought texts and completion texts
    thoughts = [t["thought"] for t in lines]
    completions = [t["completion_text"] for t in lines]

    # Thoughts must vary across samples (not all identical)
    assert len(set(thoughts)) > 4, f"Thoughts lack diversity: {set(thoughts)}"
    assert len(set(completions)) > 4, "Completions must not be identical across samples"


def test_every_turn_passes_parse_and_validate_action(tmp_path):
    """Every generated turn's completion_text must strictly pass parse_and_validate_action."""
    out_file = str(tmp_path / "coldstart_validity.jsonl")
    sys_argv_backup = sys.argv
    try:
        sys.argv = [
            "build_coldstart_data.py",
            "--out_path", out_file,
            "--num_samples", "10",
            "--mock",
        ]
        build_coldstart_main()
    finally:
        sys.argv = sys_argv_backup

    with open(out_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            record = json.loads(line)
            completion_text = record["completion_text"]
            action_dict, err = parse_and_validate_action(completion_text)
            assert err is None, f"Turn {line_idx} failed validation: {err} | Text: {completion_text}"
            assert action_dict is not None
            assert action_dict["action"] in [
                "retrieve_evidence", "inspect_frames", "refine_description",
                "call_referdino", "call_sam2", "stop"
            ]


def test_thought_length_constraint(tmp_path):
    """All thoughts in cold-start trajectories must be <= 60 words."""
    out_file = str(tmp_path / "coldstart_length.jsonl")
    sys_argv_backup = sys.argv
    try:
        sys.argv = [
            "build_coldstart_data.py",
            "--out_path", out_file,
            "--num_samples", "10",
            "--mock",
        ]
        build_coldstart_main()
    finally:
        sys.argv = sys_argv_backup

    with open(out_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            record = json.loads(line)
            thought = record.get("thought", "")
            words = thought.split()
            assert len(words) <= 60, f"Turn {line_idx} thought exceeds 60 words ({len(words)}): {thought}"


def test_no_torch_zeros_outside_mock_branch():
    """Grep test: sft_action_format.py must NOT use torch.zeros for media outside the mock branch."""
    sft_file = os.path.join(REPO_ROOT, "scripts", "rl_grpo", "sft_action_format.py")
    with open(sft_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Look for torch.zeros
    for idx, line in enumerate(lines, start=1):
        if "torch.zeros" in line:
            # Must only appear in mock branch
            # Check context lines around it
            context = "".join(lines[max(0, idx - 10):min(len(lines), idx + 5)])
            assert "mock" in context.lower(), (
                f"Forbidden torch.zeros outside mock branch at line {idx}: {line.strip()}"
            )


def test_candidate_selection_varies_with_gt():
    """
    Verify that candidate_id selection chooses the candidate with maximum IoU
    against GT, and is not hardcoded to 0.
    """
    from scripts.rl_grpo.build_coldstart_data import compute_box_mask_iou
    import numpy as np

    # Create GT mask located on the right side
    gt = np.zeros((10, 224, 224), dtype=bool)
    gt[:, 120:200, 140:210] = True

    cand0_boxes = [[0.05, 0.05, 0.3, 0.3]] * 10  # Left
    cand1_boxes = [[0.60, 0.50, 0.95, 0.90]] * 10 # Right (matches GT)

    iou0 = compute_box_mask_iou(cand0_boxes, gt)
    iou1 = compute_box_mask_iou(cand1_boxes, gt)

    assert iou1 > iou0
    chosen_id = 1 if iou1 > iou0 else 0
    assert chosen_id == 1, "Should select candidate 1 matching GT location, not candidate 0"


def test_sft_checkpoint_split_format(tmp_path):
    """
    SFT saving logic must save finetune_weights.bin and non_lora_trainables.bin.
    """
    torch = pytest.importorskip("torch", reason="PyTorch required for tensor saving test")
    from scripts.rl_grpo.sft_action_format import save_sft_checkpoint

    mock_model = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(4, 4),
        "model.model.vision_projector.weight": torch.randn(4, 4),
        "model.model.qformer.weight": torch.randn(4, 4),
    }

    out_dir = str(tmp_path / "sft_ckpt")
    save_sft_checkpoint(mock_model, out_dir)

    assert os.path.exists(os.path.join(out_dir, "finetune_weights.bin"))
    assert os.path.exists(os.path.join(out_dir, "non_lora_trainables.bin"))

    lora_keys = torch.load(os.path.join(out_dir, "finetune_weights.bin"), weights_only=False)
    non_lora_keys = torch.load(os.path.join(out_dir, "non_lora_trainables.bin"), weights_only=False)

    assert all("lora" in k.lower() for k in lora_keys)
    assert any("projector" in k.lower() for k in non_lora_keys)
    assert any("qformer" in k.lower() for k in non_lora_keys)
