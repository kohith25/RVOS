#!/usr/bin/env python3
"""
scripts/rl_grpo/check_format_validity.py - Format Validity Gate for RL Training

Evaluates whether the Ref-Thinker model satisfies the format validity gate:
valid_actions / total_actions >= 0.95

Exits with code 0 if format validity passes (>= 0.95), or code 1 if it fails.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.build_model import build_tgs_model
from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.constants import parse_and_validate_action


def evaluate_format_validity(model_adapter: TGSModelAdapter, samples: List[Dict[str, Any]], max_tokens: int = 160) -> float:
    """
    Samples policy outputs and measures the proportion of strictly valid action JSONs.
    """
    valid_count = 0
    total_count = len(samples)
    if total_count == 0:
        return 0.0

    for idx, sample in enumerate(samples):
        expression = sample.get("expression") or sample.get("ref", "")
        action = model_adapter.sample_action(
            expression=expression,
            video_tensor=sample.get("video"),
            audio_tensor=sample.get("audio"),
            history=[],
            turn_index=0,
            temperature=0.0,
            max_new_tokens=max_tokens,
        )
        if action.is_valid:
            valid_count += 1

    validity_rate = valid_count / total_count
    return validity_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ref-Thinker Action Format Validity Gate")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--avs_ckpt_dir", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_weights/Llama-2-7b-chat-hf")
    parser.add_argument("--threshold", type=float, default=0.95, help="Validity gate threshold (default 0.95)")
    parser.add_argument("--num_eval_samples", type=int, default=20)
    parser.add_argument("--mock", action="store_true", help="Run with mock model on CPU")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  Ref-Thinker Format Validity Gate Check")
    print(f"  Threshold: {args.threshold:.2%} | Samples: {args.num_eval_samples} | Mock: {args.mock}")
    print("=" * 65)

    if args.mock:
        model, _, tokenizer = build_tgs_model(use_mock=True)
        adapter = TGSModelAdapter(model=model, tokenizer=tokenizer, use_mock=True)
        eval_samples = [
            {"uid": f"mock_val_{i}", "expression": f"grounding query {i}", "video": None, "audio": None}
            for i in range(args.num_eval_samples)
        ]
    else:
        model, _, tokenizer = build_tgs_model(
            model_name_or_path=args.model_name_or_path,
            ckpt_dir=args.ckpt_dir,
            avs_ckpt_dir=args.avs_ckpt_dir,
            use_mock=False,
        )
        adapter = TGSModelAdapter(model=model, tokenizer=tokenizer, use_mock=False)
        # Real validation samples (requires dataset)
        eval_samples = [
            {"uid": f"real_val_{i}", "expression": f"target query {i}", "video": None, "audio": None}
            for i in range(args.num_eval_samples)
        ]

    validity_rate = evaluate_format_validity(adapter, eval_samples, max_tokens=args.max_new_tokens)
    print(f"[Gate Check] Validity Rate: {validity_rate:.2%} ({int(validity_rate * len(eval_samples))}/{len(eval_samples)})")

    if validity_rate < args.threshold:
        print(f"[GATE FAILED] Validity rate {validity_rate:.2%} < threshold {args.threshold:.2%}.")
        print("Model must undergo Action Format LoRA SFT before GRPO training can begin.")
        return 1

    print(f"[GATE PASSED] Model satisfies format validity requirement ({validity_rate:.2%} >= {args.threshold:.2%}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
