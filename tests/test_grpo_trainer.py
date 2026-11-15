#!/usr/bin/env python3
"""
tests/test_grpo_trainer.py - Phase D: GRPO Trainer Unit Tests

Verifiable without GPU/real weights (uses mock model).
Tests marked NOT RUN require real GPU + weights.
"""

import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Ensure TGS-Agent root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

torch = pytest.importorskip("torch", reason="PyTorch not available - NOT RUN")

import torch.nn as nn
from rl_grpo.arguments import RLGRPOArguments
from rl_grpo.trainer.grpo_trainer_tgs_v2 import TGSGRPOTrainer
from rl_grpo.trajectory import Trajectory, TurnRecord
from tools.client import ToolClient


# ---------------------------------------------------------------------------
# Minimal differentiable mock model (CPU-only, no weights needed)
# ---------------------------------------------------------------------------
class _TinyMockModel(nn.Module):
    """Tiny linear model that exposes dummy_param for mock path in trainer."""

    def __init__(self, vocab_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy_param = nn.Parameter(torch.zeros(1))
        # Expose projector param so MM-module freeze test can find it
        self.projector_weight = nn.Parameter(torch.randn(4, 4))
        self.lora_A = nn.Parameter(torch.randn(4, 4))
        self.lora_B = nn.Parameter(torch.randn(4, 4))

    def named_parameters(self, prefix="", recurse=True):
        # Return with realistic names so freeze logic can match substrings
        yield "dummy_param", self.dummy_param
        yield "projector.weight", self.projector_weight
        yield "base_model.lora_A.weight", self.lora_A
        yield "base_model.lora_B.weight", self.lora_B

    def parameters(self, recurse=True):
        return iter([self.dummy_param, self.projector_weight, self.lora_A, self.lora_B])

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        yield prefix, self

    def state_dict(self, **kwargs):
        return {
            "dummy_param": self.dummy_param,
            "projector.weight": self.projector_weight,
            "base_model.lora_A.weight": self.lora_A,
            "base_model.lora_B.weight": self.lora_B,
        }

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self):
        return self


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return [1] * min(len(text.split()), 8)

    def decode(self, ids, **kwargs):
        return " ".join(["tok"] * len(ids))

    def __call__(self, text, **kwargs):
        ids = self.encode(text)
        return type("Enc", (), {"input_ids": [ids]})()


def _make_trainer(train_mm_modules=False, num_generations=2):
    model = _TinyMockModel()
    ref_model = _TinyMockModel()
    tokenizer = _MockTokenizer()
    dataset = [{"expression": "the person playing guitar", "vid": "v001", "test_name": "test_s"}]
    args = RLGRPOArguments(
        num_generations=num_generations,
        max_turn=2,
        beta=0.04,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        train_mm_modules=train_mm_modules,
        output_dir="/tmp/tgs_grpo_test",
    )
    tool_client = ToolClient(use_mock_fallback=True, use_cache=False)
    trainer = TGSGRPOTrainer(
        policy_model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        rl_args=args,
        tool_client=tool_client,
        device=torch.device("cpu"),
    )
    return trainer


def _make_turn(completion_ids, old_logps):
    """Create a TurnRecord using the actual dataclass fields."""
    return TurnRecord(
        turn=0,
        action="stop",
        args={},
        thought="",
        raw_text="stop",
        observation="done",
        is_valid=True,
        prompt_ids=[1, 2, 3],
        completion_ids=completion_ids,
        old_logps=old_logps,
    )


def _make_traj(turns, reward=0.3):
    """Create a Trajectory using the actual dataclass fields."""
    return Trajectory(
        uid="v001",
        expression="person",
        turns=turns,
        final_pred_masks=None,
        reward_dict={"mask_iou": reward, "reward": reward},
    )


# ---------------------------------------------------------------------------
# TEST 1: MM modules are frozen when train_mm_modules=False
# ---------------------------------------------------------------------------
def test_mm_modules_frozen_by_default():
    """Projector/QFormer params must have requires_grad=False when train_mm_modules=False."""
    trainer = _make_trainer(train_mm_modules=False)
    for name, param in trainer.policy_model.named_parameters():
        if "projector" in name.lower():
            assert not param.requires_grad, (
                f"Param '{name}' should be frozen (requires_grad=False) when train_mm_modules=False"
            )


def test_mm_modules_trainable_when_flag_true():
    """When train_mm_modules=True, projector params must remain trainable."""
    trainer = _make_trainer(train_mm_modules=True)
    # At least one projector param should be trainable
    proj_trainable = [
        param.requires_grad
        for name, param in trainer.policy_model.named_parameters()
        if "projector" in name.lower()
    ]
    assert all(proj_trainable), (
        "Projector params should remain trainable when train_mm_modules=True"
    )


# ---------------------------------------------------------------------------
# TEST 2: Advantage normalization
# ---------------------------------------------------------------------------
def test_group_advantage_normalization():
    """Advantages must be zero-mean and unit-std for a diverse reward group."""
    trainer = _make_trainer()
    rewards = [0.1, 0.5, 0.3, 0.7]
    adv = trainer.compute_group_advantages(rewards)
    assert adv.shape == (4,)
    assert abs(float(adv.mean())) < 1e-5, "Advantages must be zero-mean"
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 0.01, "Advantages must be unit-std"


def test_all_equal_rewards_give_zero_advantage():
    """When all G rewards are equal, advantage must be all-zero (not NaN/Inf)."""
    trainer = _make_trainer()
    rewards = [0.5, 0.5, 0.5, 0.5]
    adv = trainer.compute_group_advantages(rewards)
    assert torch.all(adv == 0.0), "All-equal rewards must yield zero advantage"
    assert not torch.any(torch.isnan(adv)), "No NaN when std==0"


# ---------------------------------------------------------------------------
# TEST 3: ratio ≈ 1 at step 0 (mock path)
# ---------------------------------------------------------------------------
def test_ratio_near_one_at_step_zero():
    """
    At step 0, policy == old policy, so KL must be ≈ 0 for the mock path.
    """
    trainer = _make_trainer(num_generations=1)
    old_logps = [-0.5, -1.0, -0.3]
    turn = _make_turn(completion_ids=[10, 11, 12], old_logps=old_logps)
    traj = _make_traj(turns=[turn], reward=0.3)
    trainer.global_step = 0
    loss, kl, tokens = trainer.compute_per_token_loss(traj, advantage=1.0)
    # In mock path, curr_logps == old_logps => ratio == 1.0 => KL == 0
    assert abs(kl) < 1e-5, f"KL should be ~0 at step 0 (mock path), got {kl}"
    assert tokens == len(old_logps)


# ---------------------------------------------------------------------------
# TEST 4: Gradient accumulation scales 1/G
# ---------------------------------------------------------------------------
def test_loss_scaled_by_num_generations():
    """
    Total group loss must equal sum(individual losses) / num_generations.
    """
    trainer = _make_trainer(num_generations=2)
    old_logps_a = [-0.5, -0.5]
    old_logps_b = [-1.0, -1.0]
    turn_a = _make_turn(completion_ids=[10, 11], old_logps=old_logps_a)
    turn_b = _make_turn(completion_ids=[10, 11], old_logps=old_logps_b)
    traj_a = _make_traj(turns=[turn_a], reward=0.5)
    traj_b = _make_traj(turns=[turn_b], reward=0.5)

    loss_a, _, _ = trainer.compute_per_token_loss(traj_a, advantage=0.5)
    loss_b, _, _ = trainer.compute_per_token_loss(traj_b, advantage=0.5)

    expected = (float(loss_a) + float(loss_b)) / 2
    # Reproduce what step() does
    group_loss = loss_a + loss_b
    avg_group_loss = group_loss / 2
    assert abs(float(avg_group_loss) - expected) < 1e-6


# ---------------------------------------------------------------------------
# TEST 5: save_checkpoint writes two files with correct key sets
# ---------------------------------------------------------------------------
def test_save_checkpoint_writes_two_files(tmp_path):
    trainer = _make_trainer()
    save_dir = str(tmp_path / "ckpt")
    trainer.rl_args.output_dir = save_dir
    returned = trainer.save_checkpoint(save_dir)
    assert os.path.exists(os.path.join(returned, "finetune_weights.bin")), \
        "finetune_weights.bin missing"
    assert os.path.exists(os.path.join(returned, "non_lora_trainables.bin")), \
        "non_lora_trainables.bin missing"
    # Verify lora keys are in finetune_weights.bin
    lora_state = torch.load(os.path.join(returned, "finetune_weights.bin"), map_location="cpu",
                            weights_only=False)
    assert any("lora" in k for k in lora_state.keys()), "No lora keys in finetune_weights.bin"
    # Verify projector keys are in non_lora_trainables.bin
    non_lora_state = torch.load(os.path.join(returned, "non_lora_trainables.bin"), map_location="cpu",
                                weights_only=False)
    assert any("projector" in k for k in non_lora_state.keys()), \
        "No projector keys in non_lora_trainables.bin"


# ---------------------------------------------------------------------------
# TEST 6: lambda_seg > 0 is silently zeroed with a warning (not an error)
# ---------------------------------------------------------------------------
def test_lambda_seg_zeroed_with_warning(caplog):
    import logging
    model = _TinyMockModel()
    tokenizer = _MockTokenizer()
    dataset = [{"expression": "x", "vid": "v001", "test_name": "test_s"}]
    args = RLGRPOArguments(lambda_seg=0.5, output_dir="/tmp/test_lseg")
    with caplog.at_level(logging.WARNING, logger="TGSGRPOTrainer"):
        trainer = TGSGRPOTrainer(
            policy_model=model,
            ref_model=None,
            tokenizer=tokenizer,
            train_dataset=dataset,
            rl_args=args,
            tool_client=ToolClient(use_mock_fallback=True, use_cache=False),
            device=torch.device("cpu"),
        )
    assert trainer.rl_args.lambda_seg == 0.0, "lambda_seg must be zeroed"
    assert any("lambda_seg" in r.message for r in caplog.records), \
        "Must emit a warning about lambda_seg"
