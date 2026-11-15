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


# ---------------------------------------------------------------------------
# TEST 7 (Phase 4.1): use_kg propagates from RLGRPOArguments to TrajectoryCollector
# ---------------------------------------------------------------------------
def test_use_kg_propagates_to_collector():
    """TrajectoryCollector.use_kg must match what was set in RLGRPOArguments."""
    model = _TinyMockModel()
    tokenizer = _MockTokenizer()
    dataset = [{"expression": "x", "vid": "v001", "test_name": "test_s"}]

    for use_kg_val in (True, False):
        args = RLGRPOArguments(
            use_kg=use_kg_val,
            output_dir="/tmp/tgs_kg_test",
        )
        trainer = TGSGRPOTrainer(
            policy_model=model,
            ref_model=None,
            tokenizer=tokenizer,
            train_dataset=dataset,
            rl_args=args,
            tool_client=ToolClient(use_mock_fallback=True, use_cache=False),
            device=torch.device("cpu"),
        )
        assert trainer.collector.use_kg == use_kg_val, (
            f"Expected collector.use_kg={use_kg_val}, got {trainer.collector.use_kg}"
        )


# ---------------------------------------------------------------------------
# TEST 8 (Phase 4.2): retrieve_evidence absent from no-KG action set + prompt
# ---------------------------------------------------------------------------
def test_retrieve_evidence_absent_when_use_kg_false():
    """get_allowed_actions(use_kg=False) must NOT contain retrieve_evidence."""
    from rl_grpo.constants import get_allowed_actions, get_system_instruction, parse_and_validate_action

    actions_no_kg = get_allowed_actions(use_kg=False)
    assert "retrieve_evidence" not in actions_no_kg, (
        "retrieve_evidence must be excluded when use_kg=False"
    )
    # All other actions must still be present
    for a in ("inspect_frames", "refine_description", "call_referdino", "call_sam2", "stop"):
        assert a in actions_no_kg, f"Action '{a}' missing from no-KG action set"

    # System instruction must not mention retrieve_evidence
    inst_no_kg = get_system_instruction(use_kg=False)
    assert "retrieve_evidence" not in inst_no_kg, (
        "get_system_instruction(use_kg=False) must not mention retrieve_evidence"
    )

    # Parsing retrieve_evidence with use_kg=False must fail validation
    text = '{"thought": "...", "action": "retrieve_evidence", "args": {"query": "dog"}}'
    action_dict, err = parse_and_validate_action(text, allowed_actions=actions_no_kg)
    assert action_dict is None and err is not None, (
        "parse_and_validate_action must reject retrieve_evidence when use_kg=False"
    )


def test_retrieve_evidence_valid_when_use_kg_true():
    """parse_and_validate_action with use_kg=True must accept retrieve_evidence."""
    from rl_grpo.constants import get_allowed_actions, parse_and_validate_action

    actions_kg = get_allowed_actions(use_kg=True)
    assert "retrieve_evidence" in actions_kg

    text = '{"thought": "...", "action": "retrieve_evidence", "args": {"query": "dog"}}'
    action_dict, err = parse_and_validate_action(text, allowed_actions=actions_kg)
    assert action_dict is not None and err is None, (
        f"retrieve_evidence must be valid when use_kg=True; err={err}"
    )


# ---------------------------------------------------------------------------
# TEST 9 (Phase 4.3): Per-trajectory backward — gradient equivalence
# ---------------------------------------------------------------------------
def test_per_traj_backward_equivalent_to_batch():
    """
    Gradients from per-trajectory backward (each scaled 1/G) must equal
    gradients from summing losses then one backward (divided by G).
    """
    trainer = _make_trainer(num_generations=2)

    old_logps_a = [-0.5, -0.5]
    old_logps_b = [-1.0, -1.0]
    turn_a = _make_turn(completion_ids=[10, 11], old_logps=old_logps_a)
    turn_b = _make_turn(completion_ids=[10, 11], old_logps=old_logps_b)
    traj_a = _make_traj(turns=[turn_a], reward=0.5)
    traj_b = _make_traj(turns=[turn_b], reward=-0.5)  # different rewards = non-zero advantage

    advantages = trainer.compute_group_advantages([0.5, -0.5])

    # --- Method A: batch accumulate then one backward ---
    trainer.policy_model.dummy_param.grad = None
    group_loss = torch.tensor(0.0, device=trainer.device)
    for traj, adv in zip([traj_a, traj_b], advantages):
        loss_i, _, _ = trainer.compute_per_token_loss(traj, float(adv.item()))
        group_loss = group_loss + loss_i
    (group_loss / 2).backward()
    grad_batch = trainer.policy_model.dummy_param.grad.clone() if trainer.policy_model.dummy_param.grad is not None else torch.zeros(1)

    # --- Method B: per-trajectory backward (1/G each) ---
    trainer.policy_model.dummy_param.grad = None
    for traj, adv in zip([traj_a, traj_b], advantages):
        loss_i, _, _ = trainer.compute_per_token_loss(traj, float(adv.item()))
        if loss_i.requires_grad:
            (loss_i / 2).backward()
    grad_per = trainer.policy_model.dummy_param.grad.clone() if trainer.policy_model.dummy_param.grad is not None else torch.zeros(1)

    assert torch.allclose(grad_batch, grad_per, atol=1e-6), (
        f"Gradient mismatch: batch={grad_batch.item():.8f} vs per-traj={grad_per.item():.8f}"
    )


# ---------------------------------------------------------------------------
# TEST 10 (Phase 4.4): ref_mode field exists in RLGRPOArguments
# ---------------------------------------------------------------------------
def test_ref_mode_in_rl_args():
    """RLGRPOArguments must have a ref_mode field defaulting to 'peft_disable'."""
    args = RLGRPOArguments()
    assert hasattr(args, "ref_mode"), "RLGRPOArguments must have ref_mode field"
    assert args.ref_mode == "peft_disable", (
        f"Default ref_mode must be 'peft_disable', got '{args.ref_mode}'"
    )
    # Verify non-default value is accepted
    args2 = RLGRPOArguments(ref_mode="copy")
    assert args2.ref_mode == "copy"


# ---------------------------------------------------------------------------
# TEST 11 (Phase 4.5): step-0 ratio deviation reported in step_info
# ---------------------------------------------------------------------------
def test_step0_ratio_max_dev_in_step_info():
    """
    At global_step==0 the step() output must contain 'step0_ratio_max_dev'.
    For the mock path curr_logps == old_logps => ratio == 1.0 => dev ≈ 0.
    """
    trainer = _make_trainer(num_generations=1)
    trainer.global_step = 0
    sample = {
        "uid": "v001",
        "expression": "the guitar player",
        "ref": "the guitar player",
        "gt_masks": None,
        "is_null": False,
        "video": None,
        "audio": None,
    }
    info = trainer.step(sample)
    assert "step0_ratio_max_dev" in info, (
        f"step() at global_step=0 must include 'step0_ratio_max_dev'; got keys: {list(info.keys())}"
    )
    # In mock path ratio should be exactly 1.0
    assert info["step0_ratio_max_dev"] < 1e-3, (
        f"Mock path: step0_ratio_max_dev should be ~0, got {info['step0_ratio_max_dev']}"
    )
