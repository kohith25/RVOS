from __future__ import annotations
#!/usr/bin/env python3
"""
scripts/rl_grpo/train_rl_tgs.py - Entry-Point for GRPO Training on Ref-Thinker

Fixes P1:
1. Replaces nonexistent imports with `rl_grpo.build_model.build_tgs_model`.
2. Connects true dataset collator via `dataset.unified_dataset.get_dataset_collator`
   wrapped in `TGSRLDataset`.
3. Deleted broken `use_rl=False` branch (all baseline dispatch resides in `run_pipeline.py`).
4. Defaults to single GPU or CPU test execution without requiring torchrun 8-process spawns.
"""

import os
import sys
import json
import argparse
import datetime
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl_grpo.arguments import RLGRPOArguments
from rl_grpo.build_model import build_tgs_model
from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.dataset_adapter import TGSRLDataset
from rl_grpo.trainer.grpo_trainer_tgs_v2 import TGSGRPOTrainer
from tools.client import get_tool_client
from scripts.rl_grpo.check_format_validity import evaluate_format_validity


def get_parser():
    parser = argparse.ArgumentParser(description="TGS-Agent GRPO Training Entry Point")
    parser.add_argument("--ckpt_dir", type=str, default="./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/checkpoint-551",
                        help="Path to SFT checkpoint")
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_weights/Llama-2-7b-chat-hf",
                        help="Path to pretrained base LLaMA weights")
    parser.add_argument("--out_dir", type=str, default="./work_dirs_RL/tgs_grpo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--max_steps", type=int, default=100, help="Max GRPO steps")
    parser.add_argument("--num_generations", type=int, default=4, help="Group size G for GRPO")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty beta")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--grounder", type=str, default="gdino", help="Grounder tool")
    parser.add_argument("--use_kg", type=str, default="true", help="Whether to use KG")
    parser.add_argument("--skip_format_gate", action="store_true", help="Skip format validity gate check")
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Max new tokens per action turn")
    parser.add_argument("--train_mm_modules", type=str, default="false", help="Whether to train multimodal modules")
    parser.add_argument("--use_mock", action="store_true", help="Run with mock models on CPU for testing")
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args)


class MockDataset:
    """Mock dataset for quick unit testing or environments without dataset downloads."""
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "uid": f"mock_video_{idx:03d}",
            "ref": "the person playing guitar on the left",
            "instruction": "This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\nGiven the referential expression: 'the person playing guitar on the left', reason and ground.",
            "video": torch.zeros((10, 3, 224, 224), dtype=torch.float32),
            "audio": torch.zeros((10, 98, 128), dtype=torch.float32),
            "gt_masks": (np := __import__("numpy")).zeros((10, 224, 224), dtype=bool),
            "is_null": False,
        }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Normalise bool flags
    use_kg = args.use_kg.lower() in ("true", "1", "yes")
    train_mm = args.train_mm_modules.lower() in ("true", "1", "yes")

    print("=" * 65)
    print("  Starting TGS-Agent Multi-Turn RL Training (GRPO)")
    print(f"  Device: {device} | Group Size (G): {args.num_generations} | Max Steps: {args.max_steps}")
    print(f"  use_kg: {use_kg} | grounder: {args.grounder}")
    print("=" * 65)

    # 1. Build Policy Model, Reference Model, and Tokenizer
    is_mock = args.use_mock
    if is_mock:
        print("[Notice] Running explicitly in Mock mode (--use_mock passed).")

    policy_model, ref_model, tokenizer = build_tgs_model(
        model_name_or_path=args.model_name_or_path,
        ckpt_dir=args.ckpt_dir,
        device=str(device),
        use_mock=is_mock
    )

    # Write rl_meta.json so infer_pipeline can detect kg/grounder mismatch
    import json
    import datetime
    rl_meta = {
        "use_kg": use_kg,
        "grounder": args.grounder,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.out_dir, "rl_meta.json"), "w") as fh:
        json.dump(rl_meta, fh, indent=2)
    print(f"[Info] rl_meta.json written to {args.out_dir}")

    # 2. Build Dataset
    meta_csv_path = "./R2AVSBench/RefAVSBench_metadata.csv"
    if os.path.exists(meta_csv_path) and not is_mock:
        from configs.unified_config import DataArguments
        from dataset.unified_dataset import get_dataset_collator
        data_args = DataArguments()
        data_args.ref_avs_task = True
        base_dataset, _ = get_dataset_collator(data_args, tokenizer, mode="train")
        train_dataset = TGSRLDataset(base_dataset)
    else:
        print("[Notice] Using Mock Dataset for training.")
        train_dataset = MockDataset(size=20)

    # 3. Format Validity Gate Check
    if not args.skip_format_gate:
        print("[Format Gate] Evaluating model action format validity before starting GRPO...")
        adapter = TGSModelAdapter(model=policy_model, tokenizer=tokenizer, use_mock=is_mock)
        val_samples = [train_dataset[i] for i in range(min(10, len(train_dataset)))]
        valid_rate = evaluate_format_validity(adapter, val_samples, max_tokens=args.max_new_tokens)
        print(f"[Format Gate] Validity rate: {valid_rate:.2%}")
        if valid_rate < 0.95:
            raise RuntimeError(
                f"Model failed format validity gate ({valid_rate:.2%} < 95%). "
                "Must undergo Action Format LoRA SFT before GRPO training can begin. "
                "Pass --skip_format_gate to bypass."
            )

    # 4. Setup GRPO Arguments — propagate use_kg, ref_mode, train_mm_modules
    rl_args = RLGRPOArguments(
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        beta=args.beta,
        output_dir=args.out_dir,
        seed=args.seed,
        use_kg=use_kg,
        train_mm_modules=train_mm,
    )

    # 5. Initialize Trainer and Train — pass use_kg to collector via trainer
    tool_client = get_tool_client(use_mock_fallback=is_mock)
    trainer = TGSGRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        rl_args=rl_args,
        tool_client=tool_client,
        device=device
    )

    step_logs = trainer.train()
    print("\n[SUCCESS] Completed GRPO training.")


if __name__ == "__main__":
    main()
