#!/usr/bin/env python3
"""
scripts/run_pipeline.py - Unified Entry Point for TGS-Agent Pipeline

Usage:
  # Original pipeline (reproduces author's pipeline end-to-end):
  python scripts/run_pipeline.py --use_rl false --stage all --test_name test_s --benchmark refavs ...

  # RL pipeline (GRPO multi-turn policy + tool server):
  python scripts/run_pipeline.py --use_rl true --stage all --test_name test_s --benchmark refavs ...
"""

import os
import sys
import argparse
import json
import re
import uuid
import subprocess
import shutil
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ('true', '1', 'yes', 'y', 't'):
            return True
        elif val in ('false', '0', 'no', 'n', 'f'):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Entry Point for TGS-Agent (Original & RL GRPO)")

    # Core mode flag
    parser.add_argument("--use_rl", type=parse_bool, required=True,
                        help="Whether to run RL-integrated pipeline (true) or original TGS pipeline (false)")

    # Pipeline execution stage
    parser.add_argument("--stage", type=str, choices=["train", "infer", "eval", "all"], default="all",
                        help="Execution stage: train, infer, eval, or all")

    # Dataset & benchmark
    parser.add_argument("--test_name", type=str, choices=["test_s", "test_u", "test_n"], default="test_s",
                        help="Test split: test_s (seen), test_u (unseen), test_n (null)")
    parser.add_argument("--benchmark", type=str, choices=["refavs", "r2avs"], default="refavs",
                        help="Benchmark to evaluate on: refavs (Ref-AVSBench) or r2avs (R2-AVSBench)")

    # Checkpoints & directories
    parser.add_argument("--ckpt_dir", type=str, default="./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05/checkpoint-551",
                        help="Path to SFT / fine-tuned checkpoint")
    parser.add_argument("--avs_ckpt_dir", type=str, default="./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05",
                        help="Path to base checkpoint directory")
    parser.add_argument("--out_dir", type=str, default="./results/run_output",
                        help="Output directory for predictions, logs, and results.json")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Environment isolation
    parser.add_argument("--conda_env_think", type=str, default=None,
                        help="Name of conda environment for Think stage (Ref-Thinker), e.g., 'think'")
    parser.add_argument("--conda_env_dino", type=str, default=None,
                        help="Name of conda environment for Dino stage (GroundingDINO/SAM2), e.g., 'dino'")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print command line strings without executing them")

    # Ground/Segment parameters
    parser.add_argument("--box_threshold", type=float, default=0.1, help="GroundingDINO box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO text threshold")
    parser.add_argument("--use_f_object", type=parse_bool, default=False,
                        help="Whether to ground using f_object instead of s_object")
    parser.add_argument("--save_pred_mask", type=parse_bool, default=False,
                        help="Whether to save visualized predicted masks to disk")
    parser.add_argument("--media_dir", type=str, default=None, help="Path to media directory (optional override)")
    parser.add_argument("--gt_mask_dir", type=str, default=None, help="Path to ground truth masks (optional override)")
    parser.add_argument("--meta_csv_path", type=str, default=None, help="Path to metadata CSV (optional override)")

    # LLM / multimodal model args for inference
    parser.add_argument("--llm_name", type=str, default="llama", help="LLM backbone architecture")
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_weights/Llama-2-7b-chat-hf",
                        help="Base LLM pretrained weights path")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch compute device")

    # RL-specific attribution flags (strictly rejected when use_rl=false)
    parser.add_argument("--grounder", type=str, choices=["gdino", "referdino"], default="gdino",
                        help="Grounding model: gdino or referdino (only valid when use_rl=true)")
    parser.add_argument("--use_kg", type=parse_bool, default=False,
                        help="Whether to enable multimodal KG retrieval (only valid when use_rl=true)")
    parser.add_argument("--no_rl_ckpt", action="store_true",
                        help="Evaluate agent using SFT weights without RL checkpoint (for attribution)")
    parser.add_argument("--tool_server_url", type=str, default="http://127.0.0.1:8088",
                        help="Tool server URL for RL policy tool invocations")
    parser.add_argument("--mock", action="store_true",
                        help="Explicit flag to run with mock models/tools for testing on CPU")

    args = parser.parse_args()

    # Resolve all user-supplied paths against launch cwd
    if args.out_dir is not None:
        args.out_dir = os.path.abspath(args.out_dir)
    if args.ckpt_dir is not None:
        args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    if args.avs_ckpt_dir is not None:
        args.avs_ckpt_dir = os.path.abspath(args.avs_ckpt_dir)
    if args.media_dir is not None:
        args.media_dir = os.path.abspath(args.media_dir)
    if args.gt_mask_dir is not None:
        args.gt_mask_dir = os.path.abspath(args.gt_mask_dir)
    if args.meta_csv_path is not None:
        args.meta_csv_path = os.path.abspath(args.meta_csv_path)

    return args


def validate_args(args):
    # Attribution flags sanity check
    if not args.use_rl:
        if args.grounder != "gdino":
            raise ValueError(
                f"Flag --grounder={args.grounder} is only permitted when --use_rl true. "
                "The original pipeline (--use_rl false) strictly uses GroundingDINO."
            )
        if args.use_kg:
            raise ValueError(
                "Flag --use_kg true is only permitted when --use_rl true. "
                "The original pipeline (--use_rl false) does not use Knowledge Graph retrieval."
            )


def build_command_prefix(conda_env=None):
    if conda_env:
        return ["conda", "run", "-n", conda_env, "--no-capture-output", "python"]
    return [sys.executable]


def run_command_or_dry_run(cmd, dry_run=False, cwd=None, env=None):
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"[DRY RUN] {cmd_str}")
        return 0
    print(f"[EXEC] {cmd_str}")
    process = subprocess.run(cmd, cwd=cwd, env=env)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {cmd_str}")
    return process.returncode


def rewrite_ground_segment_script(original_script_path, out_temp_script_path, replacements):
    """
    Reads original ground_segment script, applies regex replacements for module-level constants,
    and writes to a temporary file path. The original file is untouched and remains byte-identical.
    """
    with open(original_script_path, "r", encoding="utf-8") as f:
        content = f.read()

    for pattern, replacement in replacements.items():
        content, count = re.subn(pattern, replacement, content)
        if count == 0:
            print(f"[WARNING] Pattern {pattern!r} was not matched in {original_script_path}")

    # Ensure ipdb set_trace() does not hang headless batch evaluations
    content = re.sub(r'^\s*set_trace\(\)', '# set_trace()', content, flags=re.MULTILINE)

    with open(out_temp_script_path, "w", encoding="utf-8") as f:
        f.write(content)


def run_use_rl_false(args):
    """
    Executes the original TGS-Agent pipeline byte-identically.
    Think -> GroundingDINO -> SAM2 -> avss_utils evaluation
    """
    os.makedirs(args.out_dir, exist_ok=True)

    # FIX 2(a, b): Canonical naming for the jsonl folder so split('_')[4:6] == [test, s|u|n]
    dir_name = f"inference_cot_ref_avs_{args.test_name}_bs"
    expected_out_jsonl_dir = os.path.join(args.out_dir, dir_name)
    os.makedirs(expected_out_jsonl_dir, exist_ok=True)
    jsonl_output_path = os.path.join(expected_out_jsonl_dir, "inference_results.jsonl")

    # Stage 1: Think (Inference via Ref-Thinker)
    if args.stage in ("infer", "all"):
        print("\n=== [Original TGS-Agent] Running Think Phase (Ref-Thinker Inference) ===")
        think_cmd = build_command_prefix(args.conda_env_think) + [
            os.path.join("scripts", "finetune", "inference_hyper_lora.py"),
            "--llm_name", args.llm_name,
            "--model_name_or_path", args.model_name_or_path,
            "--freeze_backbone", "True",
            "--lora_enable", "True",
            "--use_hyper_lora", "False",
            "--use_process", "True",
            "--bits", "32",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--lora_dropout", "0.05",
            "--bf16", "True",
            "--tf32", "False",
            "--fp16", "False",
            "--ckpt_dir", args.ckpt_dir,
            "--avqa_task", "False",
            "--ave_task", "False",
            "--avvp_task", "False",
            "--arig_task", "False",
            "--avcap_task", "False",
            "--ms3_task", "False",
            "--s4_task", "False",
            "--avss_task", "False",
            "--ref_avs_task", "True",
            "--avs_ckpt_dir", args.avs_ckpt_dir,
            "--test_name", args.test_name,
            "--device", args.device,
            "--multi_frames", "False",
            "--visual_branch", "True",
            "--video_frame_nums", "10",
            "--vit_ckpt_path", "./pretrained_weights/clip-vit-large-patch14",
            "--select_feature", "patch",
            "--image_size", "224",
            "--patch_size", "14",
            "--visual_query_token_nums", "32",
            "--audio_branch", "True",
            "--BEATs_ckpt_path", "./pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            "--audio_query_token_nums", "32",
            "--seg_branch", "False",
            "--prompt_embed_dim", "256",
            "--mask_decoder_transformer_depth", "2",
            "--low_res_mask_size", "112",
            "--image_scale_nums", "2",
            "--token_nums_per_scale", "3",
            "--avs_query_num", "300",
            "--num_classes", "1",
            "--query_generator_num_layers", "2",
            "--output_dir", args.out_dir,
        ]
        run_command_or_dry_run(think_cmd, dry_run=args.dry_run, cwd=REPO_ROOT)

        # FIX 2(a): If not dry run, check for jsonl at ckpt_dir and copy to out_dir (ALWAYS overwrite stale)
        if not args.dry_run:
            orig_think_jsonl = os.path.join(args.ckpt_dir, dir_name, "inference_results.jsonl")
            if os.path.exists(orig_think_jsonl):
                shutil.copyfile(orig_think_jsonl, jsonl_output_path)

    # Stage 2: Ground & Segment & Evaluate
    if args.stage in ("eval", "all"):
        print("\n=== [Original TGS-Agent] Running Ground/Segment Phase (GroundingDINO + SAM2) ===")
        orig_think_jsonl = os.path.join(args.ckpt_dir, dir_name, "inference_results.jsonl")

        # Copy from ckpt_dir if missing in out_dir
        if not os.path.exists(jsonl_output_path) and os.path.exists(orig_think_jsonl):
            shutil.copyfile(orig_think_jsonl, jsonl_output_path)

        if not args.dry_run and not os.path.exists(jsonl_output_path):
            raise FileNotFoundError(
                f"Inference JSONL not found at '{jsonl_output_path}' or '{orig_think_jsonl}'. "
                "Run with --stage infer first or ensure the checkpoint contains inference_results.jsonl."
            )

        if args.benchmark == "refavs":
            orig_ground_script_rel = os.path.join("ground_segment_scripts", "ground_segment_with_object_text_after_thinking_for_RefAVSBench.py")
        else:
            orig_ground_script_rel = os.path.join("ground_segment_scripts", "ground_segment_with_object_text_after_thinking_for_R2AVSBench.py")

        orig_ground_script = os.path.join(REPO_ROOT, orig_ground_script_rel)

        # Standardize paths with forward slashes for cross-platform compatibility
        norm_jsonl = jsonl_output_path.replace("\\", "/")

        replacements = {
            r'JSONL_PATH\s*=\s*["\'].*?["\']': f'JSONL_PATH = "{norm_jsonl}"',
            r'BOX_THRESHOLD\s*=\s*[0-9.]+': f'BOX_THRESHOLD = {args.box_threshold}',
            r'TEXT_THRESHOLD\s*=\s*[0-9.]+': f'TEXT_THRESHOLD = {args.text_threshold}',
            r'USE_F_OBJECT\s*=\s*(True|False)': f'USE_F_OBJECT = {args.use_f_object}',
            r'SAVE_PRED_MASK\s*=\s*(True|False)': f'SAVE_PRED_MASK = {args.save_pred_mask}',
        }

        # For data paths the user did not pass (None), do NOT add regex replacements, preserving original constants
        if args.media_dir is not None:
            norm_media = args.media_dir.replace("\\", "/")
            replacements[r'MEDIA_DIR\s*=\s*["\'].*?["\']'] = f'MEDIA_DIR = "{norm_media}"'
        if args.gt_mask_dir is not None:
            norm_gt = args.gt_mask_dir.replace("\\", "/")
            replacements[r'GT_MASK_DIR\s*=\s*["\'].*?["\']'] = f'GT_MASK_DIR = "{norm_gt}"'
        if args.meta_csv_path is not None:
            norm_meta = args.meta_csv_path.replace("\\", "/")
            replacements[r'META_CSV_PATH\s*=\s*["\'].*?["\']'] = f'META_CSV_PATH = "{norm_meta}"'

        temp_script_name = f"_tmp_eval_{args.benchmark}_{args.test_name}_{uuid.uuid4().hex[:8]}.py"
        temp_script_path = os.path.join(REPO_ROOT, "ground_segment_scripts", temp_script_name)

        # FIX: In dry_run mode, write NO files to disk!
        if args.dry_run:
            print(f"[DRY RUN] Would rewrite {orig_ground_script} into temporary script with replacements:")
            for pat, val in replacements.items():
                print(f"  {pat} => {val}")
            dino_cmd = build_command_prefix(args.conda_env_dino) + [temp_script_name]
            ground_cwd = os.path.join(REPO_ROOT, "ground_segment_scripts")
            run_command_or_dry_run(dino_cmd, dry_run=True, cwd=ground_cwd)
            return

        try:
            rewrite_ground_segment_script(orig_ground_script, temp_script_path, replacements)

            # FIX 2(c): Run in ground_segment_scripts with PYTHONPATH including repo root and Grounded-SAM-2
            gsam2_path = os.path.join(REPO_ROOT, "ground_segment_scripts", "Grounded-SAM-2")
            env = os.environ.copy()
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{gsam2_path}{os.pathsep}{existing_pp}" if existing_pp else f"{REPO_ROOT}{os.pathsep}{gsam2_path}"

            dino_cmd = build_command_prefix(args.conda_env_dino) + [temp_script_name]
            ground_cwd = os.path.join(REPO_ROOT, "ground_segment_scripts")
            run_command_or_dry_run(dino_cmd, dry_run=False, cwd=ground_cwd, env=env)

            # Locate the generated metric json
            tag_name = "f_object" if args.use_f_object else "s_object"
            expected_metric_name = f"{tag_name}_eval_results_v2_boxthre{args.box_threshold}_textthre{args.text_threshold}.json"
            if args.benchmark == "r2avs":
                expected_metric_name = f"{tag_name}_eval_results_v2_boxthre{args.box_threshold}_textthre{args.text_threshold}_reproduce.json"

            metric_json_path = os.path.join(expected_out_jsonl_dir, expected_metric_name)

            if os.path.exists(metric_json_path):
                with open(metric_json_path, "r", encoding="utf-8") as f:
                    raw_metrics = json.load(f)

                # FIX 2(d): Metric mapping for test_n vs standard splits
                per_video = {k: v for k, v in raw_metrics.items() if k not in ("overall_average_iou", "overall_average_fscore")}
                raw_iou = float(raw_metrics.get("overall_average_iou", 0.0))

                if args.test_name == "test_n":
                    standard_results = {
                        "pipeline": "original_tgs",
                        "use_rl": False,
                        "benchmark": args.benchmark,
                        "test_name": args.test_name,
                        "metrics": {
                            "J": None,
                            "F": None,
                            "J_and_F": None,
                            "S_null": round(raw_iou, 4)
                        },
                        "rl_stats": None,
                        "per_video": per_video
                    }
                else:
                    raw_f = float(raw_metrics.get("overall_average_fscore", 0.0))
                    j_and_f = (raw_iou + raw_f) / 2.0 if raw_f >= 0 else raw_iou
                    standard_results = {
                        "pipeline": "original_tgs",
                        "use_rl": False,
                        "benchmark": args.benchmark,
                        "test_name": args.test_name,
                        "metrics": {
                            "J": round(raw_iou, 4),
                            "F": round(raw_f, 4),
                            "J_and_F": round(j_and_f, 4),
                            "S_null": 0.0
                        },
                        "rl_stats": None,
                        "per_video": per_video
                    }

                standard_out_file = os.path.join(args.out_dir, "results.json")
                with open(standard_out_file, "w", encoding="utf-8") as sf:
                    json.dump(standard_results, sf, indent=2)
                print(f"[SUCCESS] Wrote standardized evaluation schema to: {standard_out_file}")

        finally:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)


def run_use_rl_true(args):
    """
    Executes the RL-integrated pipeline:
    Multi-turn RL-trained Think-agent -> KG retrieval -> ReferDINO -> candidate selection -> SAM2 refinement.
    """
    os.makedirs(args.out_dir, exist_ok=True)
    print("\n=== [RL GRPO Pipeline] Starting Execution ===")

    if args.stage in ("train", "all"):
        print("--> RL GRPO Stage: Train")
        train_cmd = build_command_prefix(args.conda_env_think) + [
            os.path.join("scripts", "rl_grpo", "train_rl_tgs.py"),
            "--ckpt_dir", args.ckpt_dir,
            "--out_dir", args.out_dir,
            "--seed", str(args.seed),
            "--grounder", args.grounder,
            "--use_kg", str(args.use_kg),
        ]
        if args.mock:
            train_cmd.append("--mock")
        run_command_or_dry_run(train_cmd, dry_run=args.dry_run, cwd=REPO_ROOT)

    if args.stage in ("infer", "eval", "all"):
        print("--> RL GRPO Stage: Multi-turn Rollout & Evaluation")
        infer_cmd = build_command_prefix(args.conda_env_think) + [
            os.path.join("scripts", "rl_grpo", "infer_pipeline.py"),
            "--ckpt_dir", args.ckpt_dir,
            "--out_dir", args.out_dir,
            "--benchmark", args.benchmark,
            "--test_name", args.test_name,
            "--grounder", args.grounder,
            "--use_kg", str(args.use_kg),
            "--tool_server_url", args.tool_server_url,
        ]
        if args.no_rl_ckpt:
            infer_cmd.append("--no_rl_ckpt")
        if args.mock:
            infer_cmd.append("--mock")
        run_command_or_dry_run(infer_cmd, dry_run=args.dry_run, cwd=REPO_ROOT)


def main():
    args = parse_args()
    validate_args(args)

    if not args.use_rl:
        run_use_rl_false(args)
    else:
        run_use_rl_true(args)


if __name__ == "__main__":
    main()
