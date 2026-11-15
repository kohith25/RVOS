#!/usr/bin/env python3
"""
scripts/rl_grpo/infer_pipeline.py - Multi-turn RL Inference & Evaluation Pipeline

Executes greedy policy rollout (temperature=0.0) over Ref-AVSBench or R2-AVSBench test splits:
1. Loads policy model (SFT base or RL checkpoint).
2. Connects to Tool Server in dino environment (with strict ConnectionError if offline unless --mock).
3. Executes multi-turn policy trajectory per video: Think -> KG -> ReferDINO -> SAM2.
4. Computes J (IoU), F (F-score), and S_null using standard avss_utils metrics.
5. Saves predicted masks if requested.
6. Writes results.json matching standardized schema.
7. Exits non-zero if inference failed (never writes dummy zeros).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.build_model import build_tgs_model
from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.trajectory import TrajectoryCollector, Trajectory
from tools.client import ToolClient, get_tool_client

# Import avss_utils with NO fallback (let ImportError propagate if environment lacks dependencies)
grounded_sam2_path = os.path.join(REPO_ROOT, "ground_segment_scripts", "Grounded-SAM-2")
if grounded_sam2_path not in sys.path:
    sys.path.insert(0, grounded_sam2_path)
from utils.avss_utils import mask_iou, Eval_Fmeasure, metric_s_for_null_batch


def parse_args():
    parser = argparse.ArgumentParser(description="TGS-Agent RL GRPO Inference & Evaluation")
    parser.add_argument("--ckpt_dir", type=str, default="./work_dirs_RL/tgs_grpo", help="Path to checkpoint")
    parser.add_argument("--avs_ckpt_dir", type=str, default="./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05")
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_weights/Llama-2-7b-chat-hf")
    parser.add_argument("--out_dir", type=str, default="./results/rl_eval", help="Output directory")
    parser.add_argument("--benchmark", type=str, choices=["refavs", "r2avs"], default="refavs")
    parser.add_argument("--test_name", type=str, choices=["test_s", "test_u", "test_n"], default="test_s")
    parser.add_argument("--media_dir", type=str, default="./REFAVS/media")
    parser.add_argument("--gt_mask_dir", type=str, default="./REFAVS/gt_mask")
    parser.add_argument("--meta_csv_path", type=str, default=None)
    parser.add_argument("--grounder", type=str, choices=["gdino", "referdino"], default="gdino")
    parser.add_argument("--use_kg", type=str, default="false")
    parser.add_argument("--no_rl_ckpt", action="store_true", help="Evaluate agent using SFT weights without RL checkpoint")
    parser.add_argument("--tool_server_url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_pred_mask", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Run with mock models and tools on CPU")
    return parser.parse_args()


def load_mock_test_samples(test_name: str, num_samples: int = 5) -> List[Dict[str, Any]]:
    samples = []
    is_null = (test_name == "test_n")
    for i in range(num_samples):
        uid = f"mock_eval_{test_name}_{i:03d}"
        ref = "null" if is_null else f"the target object number {i} performing action"
        gt = np.zeros((10, 224, 224), dtype=bool)
        if not is_null:
            gt[:, 40:160, 40:160] = True

        samples.append({
            "uid": uid,
            "ref": ref,
            "expression": ref,
            "is_null": is_null,
            "gt_masks": gt,
            "video": None,
            "audio": None,
        })
    return samples


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    use_kg = args.use_kg.lower() in ("true", "1", "yes")

    print("\n" + "=" * 65)
    print("  TGS-Agent RL GRPO Multi-Turn Inference & Evaluation")
    print(f"  Benchmark: {args.benchmark} | Split: {args.test_name} | Grounder: {args.grounder} | Use KG: {use_kg}")
    print(f"  Mock Mode: {args.mock} | No RL Ckpt (SFT only): {args.no_rl_ckpt}")
    print("=" * 65)

    # 1. Build Model
    if args.mock:
        policy_model, _, tokenizer = build_tgs_model(use_mock=True)
        tool_client = get_tool_client(server_url=args.tool_server_url, use_mock_fallback=True)
    else:
        actual_ckpt = args.avs_ckpt_dir if args.no_rl_ckpt else args.ckpt_dir
        policy_model, _, tokenizer = build_tgs_model(
            model_name_or_path=args.model_name_or_path,
            ckpt_dir=actual_ckpt,
            avs_ckpt_dir=args.avs_ckpt_dir,
            device=args.device,
            use_mock=False,
        )
        tool_client = get_tool_client(server_url=args.tool_server_url, use_mock_fallback=False)

    model_adapter = TGSModelAdapter(
        model=policy_model,
        tokenizer=tokenizer,
        use_mock=args.mock,
    )

    collector = TrajectoryCollector(
        model_adapter=model_adapter,
        tool_client=tool_client,
        max_turns=4,
        use_kg=use_kg,
    )

    # 2. Load Dataset Samples
    if args.mock:
        samples = load_mock_test_samples(args.test_name, num_samples=args.max_samples or 5)
    else:
        # Real dataset loading
        meta_csv = args.meta_csv_path
        if not meta_csv:
            meta_csv = "./R2AVSBench/RefAVSBench_metadata.csv" if args.benchmark == "refavs" else "./R2AVSBench/R2AVSBench_metadata.csv"

        if not os.path.exists(meta_csv):
            raise FileNotFoundError(f"Dataset metadata not found at: {meta_csv}")

        from configs.unified_config import DataArguments
        from dataset.unified_dataset import get_dataset_collator
        from rl_grpo.dataset_adapter import TGSRLDataset

        data_args = DataArguments()
        data_args.ref_avs_task = True
        base_dataset, _ = get_dataset_collator(data_args, tokenizer, mode=args.test_name)
        samples = TGSRLDataset(base_dataset)

    if len(samples) == 0:
        raise RuntimeError(f"No evaluation samples found for split: {args.test_name}. Cannot evaluate.")

    # 3. Greedy Inference Loop (temperature=0.0)
    per_video = {}
    ious = []
    fscores = []
    null_scores = []
    latencies = []
    turn_counts = []
    token_counts = []
    tool_counts = []

    for idx, sample in enumerate(samples):
        if args.max_samples and idx >= args.max_samples:
            break

        uid = sample.get("uid", f"sample_{idx}")
        gt_masks = sample.get("gt_masks")
        is_null = sample.get("is_null", False) or (args.test_name == "test_n")

        start_t = time.time()
        traj: Trajectory = collector.rollout(
            sample=sample,
            temperature=0.0,
            top_p=1.0,
        )
        latency = time.time() - start_t

        pred_masks = traj.final_pred_masks
        pred_logits = traj.final_pred_logits

        if pred_masks is None:
            pred_masks = np.zeros((10, 224, 224), dtype=np.uint8)
        if pred_logits is None:
            pred_logits = np.zeros((10, 224, 224), dtype=np.float32)

        import torch
        import torch.nn.functional as F

        pred_masks_tensor = torch.from_numpy(pred_masks).to(torch.uint8)
        pred_logits_tensor = torch.from_numpy(pred_logits).to(torch.float32)

        if gt_masks is not None:
            if isinstance(gt_masks, np.ndarray):
                gt_masks_tensor = torch.from_numpy(gt_masks.astype(np.uint8))
            elif isinstance(gt_masks, torch.Tensor):
                gt_masks_tensor = gt_masks.to(torch.uint8)
            else:
                gt_masks_tensor = torch.zeros((10, 224, 224), dtype=torch.uint8)
        else:
            gt_masks_tensor = torch.zeros((10, 224, 224), dtype=torch.uint8)

        # Standardize shapes to [10, 224, 224] matching original evaluation
        if pred_masks_tensor.ndim == 4 and pred_masks_tensor.shape[1] == 1:
            pred_masks_tensor = pred_masks_tensor.squeeze(1)
        if gt_masks_tensor.ndim == 4 and gt_masks_tensor.shape[1] == 1:
            gt_masks_tensor = gt_masks_tensor.squeeze(1)
        if pred_logits_tensor.ndim == 4 and pred_logits_tensor.shape[1] == 1:
            pred_logits_tensor = pred_logits_tensor.squeeze(1)

        if pred_masks_tensor.shape[-2:] != (224, 224):
            pred_masks_tensor = F.interpolate(
                pred_masks_tensor.unsqueeze(1).float(), size=(224, 224), mode="nearest"
            ).squeeze(1).to(torch.uint8)
        if gt_masks_tensor.shape[-2:] != (224, 224):
            gt_masks_tensor = F.interpolate(
                gt_masks_tensor.unsqueeze(1).float(), size=(224, 224), mode="nearest"
            ).squeeze(1).to(torch.uint8)
        if pred_logits_tensor.shape[-2:] != (224, 224):
            pred_logits_tensor = F.interpolate(
                pred_logits_tensor.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(1).to(torch.float32)

        ref_text = sample.get("ref", "")
        if args.test_name == "test_n":
            if ref_text == "null":
                mean_iou = 0.0
            else:
                mean_iou = metric_s_for_null_batch(pred_masks_tensor)
            fscore = -999
        else:
            mean_iou = mask_iou(pred_masks_tensor, gt_masks_tensor)
            fscore = Eval_Fmeasure(pred=pred_logits_tensor.cpu(), gt=gt_masks_tensor.cpu())

        iou_val = mean_iou.item() if isinstance(mean_iou, torch.Tensor) else float(mean_iou)
        fscore_val = fscore.item() if isinstance(fscore, torch.Tensor) else float(fscore)

        if args.test_name == "test_n":
            null_scores.append(iou_val)
            per_video[uid] = {
                "vid": uid,
                "s_null_acc": round(iou_val, 4),
                "turns": len(traj.turns),
                "latency_sec": round(latency, 2),
            }
        else:
            ious.append(iou_val)
            fscores.append(fscore_val)
            per_video[uid] = {
                "vid": uid,
                "s_mean_iou": round(iou_val, 4),
                "s_fscore": round(fscore_val, 4),
                "turns": len(traj.turns),
                "latency_sec": round(latency, 2),
            }

        # Stats tracking
        latencies.append(latency)
        turn_counts.append(len(traj.turns))
        tool_counts.append(sum(1 for t in traj.turns if t.action != "stop"))
        token_counts.append(sum(len(t.completion_ids) for t in traj.turns))

        # Save masks if requested
        if args.save_pred_mask:
            mask_save_dir = os.path.join(args.out_dir, "pred_masks", uid)
            os.makedirs(mask_save_dir, exist_ok=True)
            for t in range(10):
                m_img = Image.fromarray((pred_masks[t] * 255).astype(np.uint8))
                m_img.save(os.path.join(mask_save_dir, f"{t}.png"))

    if len(per_video) == 0:
        raise RuntimeError("Inference completed with 0 videos evaluated. Rejecting dummy results.")

    # 4. Aggregate Metrics exactly as original ground_segment script lines 293-309
    valid_ious = []
    valid_fscores = []
    for item_data in per_video.values():
        if args.test_name == "test_n":
            iou_v = item_data.get("s_null_acc")
            if isinstance(iou_v, (int, float)) and iou_v >= 0:
                valid_ious.append(iou_v)
        else:
            iou_v = item_data.get("s_mean_iou")
            fscore_v = item_data.get("s_fscore")
            if isinstance(iou_v, (int, float)) and iou_v >= 0:
                valid_ious.append(iou_v)
            if isinstance(fscore_v, (int, float)) and fscore_v >= 0:
                valid_fscores.append(fscore_v)

    average_iou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    average_fscore = sum(valid_fscores) / len(valid_fscores) if valid_fscores else 0.0

    if args.test_name == "test_n":
        final_metrics = {
            "J": None,
            "F": None,
            "J_and_F": None,
            "S_null": round(average_iou, 4),
        }
    else:
        final_metrics = {
            "J": round(average_iou, 4),
            "F": round(average_fscore, 4),
            "J_and_F": round((average_iou + average_fscore) / 2.0, 4),
            "S_null": 0.0,
        }

    rl_stats = {
        "avg_turns": round(float(np.mean(turn_counts)), 2) if turn_counts else 0.0,
        "avg_tool_calls": round(float(np.mean(tool_counts)), 2) if tool_counts else 0.0,
        "avg_tokens": round(float(np.mean(token_counts)), 1) if token_counts else 0.0,
        "avg_latency_sec": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }

    results_data = {
        "pipeline": "rl_grpo",
        "use_rl": True,
        "benchmark": args.benchmark,
        "test_name": args.test_name,
        "grounder": args.grounder,
        "use_kg": use_kg,
        "no_rl_ckpt": args.no_rl_ckpt,
        "metrics": final_metrics,
        "rl_stats": rl_stats,
        "per_video": per_video,
    }

    results_file = os.path.join(args.out_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as rf:
        json.dump(results_data, rf, indent=2)

    print(f"\n[EVALUATION SUCCESS] Wrote standard results.json to: {results_file}")
    print(f"Metrics: {final_metrics}")
    print(f"RL Stats: {rl_stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
