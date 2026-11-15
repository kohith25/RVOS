#!/usr/bin/env python3
"""
tests/test_model_builder.py - Unit tests for Ref-Thinker model builder

Tests:
1. validate_checkpoint_keys accepts expected frozen backbone missing keys.
2. validate_checkpoint_keys raises on unexpected keys.
3. validate_checkpoint_keys raises on missing LoRA/projector/QFormer keys.
4. build_tgs_model raises FileNotFoundError when weights do not exist and use_mock=False.
5. build_tgs_model succeeds cleanly when use_mock=True, returning policy, ref, and tokenizer.
6. Reference model support for copy vs peft_disable mode.
"""

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.build_model import (
    validate_checkpoint_keys,
    build_tgs_model,
    MockTokenizer,
    MockThinkerModel,
)


def test_validate_checkpoint_keys_valid():
    """Missing frozen backbone keys should be allowed."""
    missing = ["model.layers.0.self_attn.q_proj.weight", "model.embed_tokens.weight"]
    unexpected = []
    # Should not raise
    validate_checkpoint_keys(missing, unexpected)


def test_validate_checkpoint_keys_unexpected():
    """Unexpected keys must raise RuntimeError."""
    missing = []
    unexpected = ["extra_unknown_layer.bias"]
    with pytest.raises(RuntimeError, match="unexpected_keys found"):
        validate_checkpoint_keys(missing, unexpected)


def test_validate_checkpoint_keys_missing_lora():
    """Missing LoRA trainable keys must raise RuntimeError."""
    missing = ["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
    unexpected = []
    with pytest.raises(RuntimeError, match="critical trainable keys are missing"):
        validate_checkpoint_keys(missing, unexpected)


def test_validate_checkpoint_keys_missing_projector():
    """Missing multimodal projector / Q-Former keys must raise RuntimeError."""
    missing = ["model.visual_projector.qformer.weight"]
    unexpected = []
    with pytest.raises(RuntimeError, match="critical trainable keys are missing"):
        validate_checkpoint_keys(missing, unexpected)


def test_build_model_raises_without_mock():
    """When weights are missing and use_mock=False, build_tgs_model must RAISE FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Pretrained LLM backbone not found"):
        build_tgs_model(
            model_name_or_path="./nonexistent_path_to_weights_12345",
            use_mock=False,
        )


def test_build_model_mock_mode():
    """Mock mode must succeed cleanly without GPU or weights."""
    policy, ref, tokenizer = build_tgs_model(use_mock=True, ref_mode="copy")
    assert policy is not None
    assert ref is not None
    assert tokenizer is not None
    assert tokenizer.pad_token_id == 0

    # Test peft_disable ref_mode
    policy_pd, ref_pd, tok_pd = build_tgs_model(use_mock=True, ref_mode="peft_disable")
    assert policy_pd is ref_pd
    with policy_pd.disable_adapter():
        assert policy_pd._disabled_adapter is True
    assert policy_pd._disabled_adapter is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
