#!/usr/bin/env python3
"""
tests/test_format_gate.py - Unit tests for Phase B Cold-Start SFT Data & Format Gate
"""

import os
import sys
import json
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.rl_grpo.build_coldstart_data import build_expert_trajectory
from scripts.rl_grpo.check_format_validity import evaluate_format_validity
from rl_grpo.build_model import build_tgs_model
from rl_grpo.model_adapter import TGSModelAdapter, ModelAction


def test_build_coldstart_trajectory():
    """Verify scripted expert trajectory generation."""
    expr = "a man playing the acoustic guitar"
    turns = build_expert_trajectory(expr, uid="sample_test", mock=True)

    # Must produce 5 sequential mock turns, including the shared inspect_frames path.
    assert len(turns) == 5
    expected_actions = ["retrieve_evidence", "inspect_frames", "call_referdino", "call_sam2", "stop"]
    for i, expected_act in enumerate(expected_actions):
        t = turns[i]
        assert t["action"] == expected_act
        assert t["mock"] is True
        # Thought word count constraint: <= 60 words
        word_count = len(t["thought"].split())
        assert word_count <= 60, f"Turn {i} thought has {word_count} words (> 60)"
        # Completion text must be valid JSON
        comp_json = json.loads(t["completion_text"])
        assert comp_json["action"] == expected_act
        assert "args" in comp_json


def test_format_validity_mock_model():
    """Verify that Mock model passes the format validity gate (>= 95%)."""
    model, _, tokenizer = build_tgs_model(use_mock=True)
    adapter = TGSModelAdapter(model=model, tokenizer=tokenizer, use_mock=True)

    samples = [
        {"uid": f"test_{i}", "expression": "dog running in park", "video": None, "audio": None}
        for i in range(20)
    ]

    validity_rate = evaluate_format_validity(adapter, samples, max_tokens=160)
    assert validity_rate >= 0.95, f"Mock model validity rate {validity_rate} < 0.95"


def test_format_validity_rejection():
    """Verify that an adapter producing invalid format is rejected (< 95%)."""
    class BadAdapter:
        def sample_action(self, *args, **kwargs):
            return ModelAction(
                action_dict=None,
                action_name=None,
                args={},
                thought="",
                raw_text="This is not valid json at all",
                is_valid=False,
                error="Invalid JSON",
                prompt_ids=[],
                completion_ids=[],
                old_logps=[]
            )

    samples = [
        {"uid": f"test_{i}", "expression": "dog running in park", "video": None, "audio": None}
        for i in range(10)
    ]

    validity_rate = evaluate_format_validity(BadAdapter(), samples)
    assert validity_rate == 0.0
    assert validity_rate < 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
