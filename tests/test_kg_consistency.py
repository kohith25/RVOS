from __future__ import annotations
#!/usr/bin/env python3
"""
tests/test_kg_consistency.py - Tests for Phase 4: Training/Inference Consistency
Runs on CPU without PyTorch.
"""

import os
import sys
import json
import pytest
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.arguments import RLGRPOArguments
from rl_grpo.constants import (
    get_allowed_actions,
    get_system_instruction,
    parse_and_validate_action,
)
from scripts.run_pipeline import get_parser as get_pipeline_parser, validate_args
from scripts.rl_grpo.infer_pipeline import parse_args as parse_infer_args


def test_ref_mode_in_rl_args():
    """Phase 4.4: RLGRPOArguments must have a ref_mode field defaulting to 'peft_disable'."""
    args = RLGRPOArguments()
    assert hasattr(args, "ref_mode"), "RLGRPOArguments must have ref_mode field"
    assert args.ref_mode == "peft_disable", (
        f"Default ref_mode must be 'peft_disable', got '{args.ref_mode}'"
    )
    args_copy = RLGRPOArguments(ref_mode="copy")
    assert args_copy.ref_mode == "copy"


def test_use_kg_default_in_rl_args():
    """Phase 4.1: RLGRPOArguments use_kg must default to True."""
    args = RLGRPOArguments()
    assert hasattr(args, "use_kg")
    assert args.use_kg is True


def test_retrieve_evidence_absent_when_use_kg_false():
    """Phase 4.2: When use_kg=False, retrieve_evidence is removed from allowed actions and system instruction."""
    actions_no_kg = get_allowed_actions(use_kg=False)
    assert "retrieve_evidence" not in actions_no_kg, (
        "retrieve_evidence must not be in allowed actions when use_kg=False"
    )

    inst_no_kg = get_system_instruction(use_kg=False)
    assert "retrieve_evidence" not in inst_no_kg, (
        "retrieve_evidence must not be advertised in system instruction when use_kg=False"
    )

    # Parsing a retrieve_evidence action with allowed_actions=actions_no_kg must fail validation
    text = '{"thought": "look up kg", "action": "retrieve_evidence", "args": {"query": "guitar"}}'
    data, err = parse_and_validate_action(text, allowed_actions=actions_no_kg)
    assert data is None
    assert err is not None
    assert "not in allowed actions" in err


def test_retrieve_evidence_present_when_use_kg_true():
    """Phase 4.2: When use_kg=True, retrieve_evidence is present and valid."""
    actions_kg = get_allowed_actions(use_kg=True)
    assert "retrieve_evidence" in actions_kg

    inst_kg = get_system_instruction(use_kg=True)
    assert "retrieve_evidence" in inst_kg

    text = '{"thought": "look up kg", "action": "retrieve_evidence", "args": {"query": "guitar"}}'
    data, err = parse_and_validate_action(text, allowed_actions=actions_kg)
    assert err is None
    assert data is not None
    assert data["action"] == "retrieve_evidence"


def test_infer_pipeline_kg_mismatch_raises(tmp_path):
    """
    Phase 4.1: infer_pipeline.py must raise ValueError if inference --use_kg differs
    from the training checkpoint metadata rl_meta.json unless --allow_kg_mismatch is passed.
    """
    # Create mock ckpt dir with rl_meta.json specifying use_kg = True
    ckpt_dir = tmp_path / "test_ckpt"
    ckpt_dir.mkdir()
    meta_file = ckpt_dir / "rl_meta.json"
    with open(meta_file, "w") as f:
        json.dump({"use_kg": True, "grounder": "gdino"}, f)

    # When infer_pipeline is run with use_kg=False on this checkpoint, it must raise ValueError
    from scripts.rl_grpo.infer_pipeline import main as infer_main
    sys_argv_backup = sys.argv
    try:
        sys.argv = [
            "infer_pipeline.py",
            "--ckpt_dir", str(ckpt_dir),
            "--use_kg", "false",
            "--out_dir", str(tmp_path / "out"),
        ]
        with pytest.raises(ValueError, match="KG mismatch: checkpoint was trained with use_kg=True"):
            infer_main()
    finally:
        sys.argv = sys_argv_backup


def test_run_pipeline_default_use_kg_consistency():
    """
    Phase 4.1: run_pipeline defaults --use_kg to True when --use_rl is true.
    """
    parser = get_pipeline_parser()

    # When --use_rl true and --use_kg not specified
    args = parser.parse_args(["--use_rl", "true"])
    validate_args(args)
    assert args.use_kg is True, "Expected use_kg to default to True when use_rl=True"

    # When --use_rl false and --use_kg not specified
    args_false = parser.parse_args(["--use_rl", "false"])
    validate_args(args_false)
    assert args_false.use_kg is False, "Expected use_kg to default to False when use_rl=False"
