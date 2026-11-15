#!/usr/bin/env python3
"""
tests/test_use_rl_false.py - Tests for Harness and use_rl=false Pipeline

Tests:
1. Rejection of attribution flags (--grounder, --use_kg) when use_rl=false
2. Acceptance of attribution flags when use_rl=true
3. Regex rewriting of module-level constants on a temp file copy (original unmodified)
4. Dry-run command construction for both think (infer) and dino (eval) stages
5. Standardized results schema creation and validation
6. Comparison harness (scripts/compare_runs.py) calculation of diffs, winners, and tables
"""

import os
import sys
import json
import uuid
import tempfile
import argparse
import pytest
from unittest.mock import patch, MagicMock

# Add TGS-Agent repo root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.run_pipeline import (
    validate_args,
    rewrite_ground_segment_script,
    run_use_rl_false,
    parse_bool,
)
from scripts.compare_runs import load_results, format_delta


class DummyArgs:
    def __init__(self, **kwargs):
        self.use_rl = kwargs.get("use_rl", False)
        self.stage = kwargs.get("stage", "all")
        self.test_name = kwargs.get("test_name", "test_s")
        self.benchmark = kwargs.get("benchmark", "refavs")
        self.ckpt_dir = kwargs.get("ckpt_dir", "checkpoints/sft")
        self.avs_ckpt_dir = kwargs.get("avs_ckpt_dir", "checkpoints")
        self.out_dir = kwargs.get("out_dir", "results/test_out")
        self.seed = kwargs.get("seed", 42)
        self.grounder = kwargs.get("grounder", "gdino")
        self.use_kg = kwargs.get("use_kg", False)
        self.conda_env_think = kwargs.get("conda_env_think", None)
        self.conda_env_dino = kwargs.get("conda_env_dino", None)
        self.dry_run = kwargs.get("dry_run", True)
        self.box_threshold = kwargs.get("box_threshold", 0.1)
        self.text_threshold = kwargs.get("text_threshold", 0.25)
        self.use_f_object = kwargs.get("use_f_object", False)
        self.save_pred_mask = kwargs.get("save_pred_mask", False)
        self.media_dir = kwargs.get("media_dir", "./REFAVS/media")
        self.gt_mask_dir = kwargs.get("gt_mask_dir", "./REFAVS/gt_mask")
        self.meta_csv_path = kwargs.get("meta_csv_path", None)
        self.llm_name = kwargs.get("llm_name", "llama")
        self.model_name_or_path = kwargs.get("model_name_or_path", "./weights")
        self.device = kwargs.get("device", "cuda:0")


def test_flag_rejection_when_use_rl_false():
    """Verify that --grounder and --use_kg are strictly rejected when use_rl=false."""
    # Disallowed grounder
    args_bad_grounder = DummyArgs(use_rl=False, grounder="referdino", use_kg=False)
    with pytest.raises(ValueError, match="only permitted when --use_rl true"):
        validate_args(args_bad_grounder)

    # Disallowed use_kg
    args_bad_kg = DummyArgs(use_rl=False, grounder="gdino", use_kg=True)
    with pytest.raises(ValueError, match="only permitted when --use_rl true"):
        validate_args(args_bad_kg)

    # Valid when use_rl=false
    args_valid = DummyArgs(use_rl=False, grounder="gdino", use_kg=False)
    validate_args(args_valid)  # Should not raise


def test_flag_acceptance_when_use_rl_true():
    """Verify that attribution flags are accepted when use_rl=true."""
    args_rl = DummyArgs(use_rl=True, grounder="referdino", use_kg=True)
    validate_args(args_rl)  # Should not raise


def test_temp_copy_regex_rewriting():
    """Verify regex replaces module-level constants in temp copy while original file is untouched."""
    orig_script = os.path.join(REPO_ROOT, "ground_segment_scripts", "ground_segment_with_object_text_after_thinking_for_RefAVSBench.py")
    assert os.path.exists(orig_script), f"Missing {orig_script}"

    with open(orig_script, "rb") as f:
        orig_bytes = f.read()

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_out_script = os.path.join(tmp_dir, "temp_ground_eval.py")

        replacements = {
            r'JSONL_PATH\s*=\s*["\'].*?["\']': 'JSONL_PATH = "custom/test/path/results.jsonl"',
            r'BOX_THRESHOLD\s*=\s*[0-9.]+': 'BOX_THRESHOLD = 0.35',
            r'TEXT_THRESHOLD\s*=\s*[0-9.]+': 'TEXT_THRESHOLD = 0.45',
            r'USE_F_OBJECT\s*=\s*(True|False)': 'USE_F_OBJECT = True',
            r'SAVE_PRED_MASK\s*=\s*(True|False)': 'SAVE_PRED_MASK = True',
            r'MEDIA_DIR\s*=\s*["\'].*?["\']': 'MEDIA_DIR = "custom/media"',
            r'GT_MASK_DIR\s*=\s*["\'].*?["\']': 'GT_MASK_DIR = "custom/gt"',
            r'META_CSV_PATH\s*=\s*["\'].*?["\']': 'META_CSV_PATH = "custom/meta.csv"',
        }

        rewrite_ground_segment_script(orig_script, temp_out_script, replacements)

        with open(temp_out_script, "r", encoding="utf-8") as f:
            new_content = f.read()

        assert 'JSONL_PATH = "custom/test/path/results.jsonl"' in new_content
        assert 'BOX_THRESHOLD = 0.35' in new_content
        assert 'TEXT_THRESHOLD = 0.45' in new_content
        assert 'USE_F_OBJECT = True' in new_content
        assert 'SAVE_PRED_MASK = True' in new_content
        assert 'MEDIA_DIR = "custom/media"' in new_content
        assert 'GT_MASK_DIR = "custom/gt"' in new_content
        assert 'META_CSV_PATH = "custom/meta.csv"' in new_content

    # Assert original file is 100% byte-identical
    with open(orig_script, "rb") as f:
        assert f.read() == orig_bytes, "Original script was modified!"


def test_dry_run_execution():
    """Verify --dry_run prints exact commands without error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = DummyArgs(
            use_rl=False,
            stage="all",
            test_name="test_s",
            benchmark="refavs",
            out_dir=tmp_dir,
            dry_run=True,
            box_threshold=0.15,
            text_threshold=0.30
        )
        # Should execute dry-run smoothly without spawning real processes
        run_use_rl_false(args)


def test_results_schema_and_comparison():
    """Verify common results.json generation and compare_runs calculations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir1 = os.path.join(tmp_dir, "run1_orig")
        dir2 = os.path.join(tmp_dir, "run2_rl")
        os.makedirs(dir1, exist_ok=True)
        os.makedirs(dir2, exist_ok=True)

        res1 = {
            "pipeline": "original_tgs",
            "use_rl": False,
            "benchmark": "refavs",
            "test_name": "test_s",
            "metrics": {"J": 0.5500, "F": 0.5200, "J_and_F": 0.5350, "S_null": 0.0},
            "rl_stats": None,
            "per_video": {
                "vid_001": {"vid": "v1", "s_mean_iou": 0.50, "s_fscore": 0.45},
                "vid_002": {"vid": "v2", "s_mean_iou": 0.60, "s_fscore": 0.55}
            }
        }
        res2 = {
            "pipeline": "rl_grpo",
            "use_rl": True,
            "benchmark": "refavs",
            "test_name": "test_s",
            "metrics": {"J": 0.6200, "F": 0.5900, "J_and_F": 0.6050, "S_null": 0.0},
            "rl_stats": {"avg_turns": 3.0, "avg_tool_calls": 3.5, "avg_tokens": 120.0, "avg_latency_sec": 1.1},
            "per_video": {
                "vid_001": {"vid": "v1", "s_mean_iou": 0.65, "s_fscore": 0.60},
                "vid_002": {"vid": "v2", "s_mean_iou": 0.58, "s_fscore": 0.57}
            }
        }

        with open(os.path.join(dir1, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res1, f)
        with open(os.path.join(dir2, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res2, f)

        # Test load_results
        loaded1 = load_results(dir1)
        loaded2 = load_results(dir2)
        assert loaded1["metrics"]["J"] == 0.55
        assert loaded2["metrics"]["J"] == 0.62

        # Test format_delta
        assert format_delta(loaded1["metrics"]["J"], loaded2["metrics"]["J"]) == "+0.0700"

        # Test compare_runs script execution via CLI
        compare_script = os.path.join(REPO_ROOT, "scripts/compare_runs.py")
        sys_argv_backup = sys.argv
        try:
            sys.argv = [
                compare_script,
                "--run1", dir1,
                "--run2", dir2,
                "--label1", "Original",
                "--label2", "RL_Model",
                "--out_dir", dir2
            ]
            import scripts.compare_runs as cr
            cr.main()
        finally:
            sys.argv = sys_argv_backup

        summary_file = os.path.join(dir2, "comparison_summary.json")
        csv_file = os.path.join(dir2, "comparison_per_video.csv")
        assert os.path.exists(summary_file)
        assert os.path.exists(csv_file)

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["counts"]["total_videos"] == 2
            assert summary["counts"]["run2_wins"] == 2  # both vid_001 and vid_002 improved in J or F


def test_subprocess_invocation_and_cwd_env():
    """
    FIX 2(a, b, c, e, f): Verify end-to-end process invocation:
    - Canonical JSONL path & TEST_NAME split logic
    - ground script cwd == ground_segment_scripts
    - PYTHONPATH contains REPO_ROOT and Grounded-SAM-2
    - Standardized results.json generation
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = os.path.join(tmp_dir, "fake_ckpt")
        out_dir = os.path.join(tmp_dir, "run_out")
        jsonl_subdir = os.path.join(ckpt_dir, "inference_cot_ref_avs_test_s_bs")
        os.makedirs(jsonl_subdir, exist_ok=True)
        fixture_jsonl = os.path.join(jsonl_subdir, "inference_results.jsonl")
        with open(fixture_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video_id": "v1", "f_object": "dog", "s_object": "barking dog"}) + "\n")

        args = DummyArgs(
            use_rl=False,
            stage="all",
            test_name="test_s",
            benchmark="refavs",
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            conda_env_think="think_env",
            conda_env_dino="dino_env",
            dry_run=False,
            box_threshold=0.1,
            text_threshold=0.25,
            use_f_object=False
        )

        calls = []

        def fake_subprocess_run(cmd, cwd=None, env=None):
            calls.append({"cmd": cmd, "cwd": cwd, "env": env})
            # When dino script is run, simulate generation of raw metric json
            if cwd and "ground_segment_scripts" in cwd:
                metric_file = os.path.join(out_dir, "inference_cot_ref_avs_test_s_bs", "s_object_eval_results_v2_boxthre0.1_textthre0.25.json")
                os.makedirs(os.path.dirname(metric_file), exist_ok=True)
                with open(metric_file, "w", encoding="utf-8") as mf:
                    json.dump({
                        "overall_average_iou": 0.6543,
                        "overall_average_fscore": 0.5891,
                        "v1": {"s_mean_iou": 0.6543, "s_fscore": 0.5891}
                    }, mf)
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res

        with patch("scripts.run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            run_use_rl_false(args)

        assert len(calls) == 2, f"Expected 2 stage executions (think, dino), got {len(calls)}"

        # Verify Stage 1 (Think)
        think_call = calls[0]
        assert think_call["cmd"][:5] == ["conda", "run", "-n", "think_env", "--no-capture-output"]
        assert think_call["cwd"] == REPO_ROOT

        # Verify Stage 2 (Dino/Ground)
        dino_call = calls[1]
        assert dino_call["cmd"][:5] == ["conda", "run", "-n", "dino_env", "--no-capture-output"]
        expected_dino_cwd = os.path.join(REPO_ROOT, "ground_segment_scripts")
        assert os.path.abspath(dino_call["cwd"]) == os.path.abspath(expected_dino_cwd)

        # Verify PYTHONPATH
        env = dino_call["env"]
        assert env is not None and "PYTHONPATH" in env
        pp = env["PYTHONPATH"]
        assert REPO_ROOT in pp
        assert os.path.join(expected_dino_cwd, "Grounded-SAM-2") in pp

        # Verify canonical jsonl was copied to out_dir
        out_jsonl = os.path.join(out_dir, "inference_cot_ref_avs_test_s_bs", "inference_results.jsonl")
        assert os.path.exists(out_jsonl)

        # Verify TEST_NAME split logic
        norm_jsonl = out_jsonl.replace("\\", "/")
        parent_dir = norm_jsonl.split("/")[-2]
        assert parent_dir == "inference_cot_ref_avs_test_s_bs"
        parsed_test_name = parent_dir.split("_")[4] + "_" + parent_dir.split("_")[5]
        assert parsed_test_name == "test_s"

        # Verify standardized results.json
        results_file = os.path.join(out_dir, "results.json")
        assert os.path.exists(results_file)
        with open(results_file, "r", encoding="utf-8") as rf:
            res_data = json.load(rf)
        assert res_data["pipeline"] == "original_tgs"
        assert res_data["use_rl"] is False
        assert res_data["metrics"]["J"] == 0.6543
        assert res_data["metrics"]["F"] == 0.5891
        assert res_data["metrics"]["J_and_F"] == round((0.6543 + 0.5891) / 2.0, 4)
        assert res_data["metrics"]["S_null"] == 0.0


def test_metric_mapping_test_n():
    """
    FIX 2(d): Verify that test_n (null audio) maps metric exclusively to S_null,
    with J and F left as None rather than meaningless 0s.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = os.path.join(tmp_dir, "fake_ckpt")
        out_dir = os.path.join(tmp_dir, "run_out_test_n")
        jsonl_subdir = os.path.join(ckpt_dir, "inference_cot_ref_avs_test_n_bs")
        os.makedirs(jsonl_subdir, exist_ok=True)
        fixture_jsonl = os.path.join(jsonl_subdir, "inference_results.jsonl")
        with open(fixture_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video_id": "v_null", "s_object": "null"}) + "\n")

        args = DummyArgs(
            use_rl=False,
            stage="eval",
            test_name="test_n",
            benchmark="refavs",
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            dry_run=False,
            box_threshold=0.1,
            text_threshold=0.25,
            use_f_object=False
        )

        def fake_subprocess_run(cmd, cwd=None, env=None):
            # Simulate metric json for test_n
            metric_file = os.path.join(out_dir, "inference_cot_ref_avs_test_n_bs", "s_object_eval_results_v2_boxthre0.1_textthre0.25.json")
            os.makedirs(os.path.dirname(metric_file), exist_ok=True)
            with open(metric_file, "w", encoding="utf-8") as mf:
                json.dump({
                    "overall_average_iou": 0.8421,
                    "overall_average_fscore": -1.0,
                    "v_null": {"s_mean_iou": 0.8421, "s_fscore": -1.0}
                }, mf)
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res

        with patch("scripts.run_pipeline.subprocess.run", side_effect=fake_subprocess_run):
            run_use_rl_false(args)

        results_file = os.path.join(out_dir, "results.json")
        assert os.path.exists(results_file)
        with open(results_file, "r", encoding="utf-8") as rf:
            res_data = json.load(rf)

        assert res_data["metrics"]["J"] is None
        assert res_data["metrics"]["F"] is None
        assert res_data["metrics"]["J_and_F"] is None
        assert res_data["metrics"]["S_null"] == 0.8421


def test_dry_run_leaves_zero_files():
    """Verify that --dry_run creates ZERO temporary files in ground_segment_scripts."""
    ground_dir = os.path.join(REPO_ROOT, "ground_segment_scripts")
    files_before = set(os.listdir(ground_dir))

    args = DummyArgs(
        use_rl=False,
        stage="eval",
        test_name="test_s",
        benchmark="refavs",
        ckpt_dir="dummy/ckpt",
        out_dir="dummy/out",
        dry_run=True,
    )
    run_use_rl_false(args)

    files_after = set(os.listdir(ground_dir))
    diff_files = files_after - files_before
    assert len(diff_files) == 0, f"Dry run left temporary files: {diff_files}"


def test_eval_stage_missing_jsonl_raises():
    """Verify that running --stage eval without JSONL at either location raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = DummyArgs(
            use_rl=False,
            stage="eval",
            test_name="test_s",
            benchmark="refavs",
            ckpt_dir=os.path.join(tmp_dir, "nonexistent_ckpt"),
            out_dir=os.path.join(tmp_dir, "nonexistent_out"),
            dry_run=False,
        )
        with pytest.raises(FileNotFoundError, match="Inference JSONL not found"):
            run_use_rl_false(args)


def test_jsonl_overwritten_on_rerun():
    """Verify that fresh inference_results.jsonl from ckpt_dir overwrites stale output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = os.path.join(tmp_dir, "ckpt")
        out_dir = os.path.join(tmp_dir, "out")
        jsonl_subdir = os.path.join(ckpt_dir, "inference_cot_ref_avs_test_s_bs")
        os.makedirs(jsonl_subdir, exist_ok=True)

        fresh_jsonl = os.path.join(jsonl_subdir, "inference_results.jsonl")
        with open(fresh_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video_id": "fresh_video"}) + "\n")

        # Create stale file in out_dir
        out_subdir = os.path.join(out_dir, "inference_cot_ref_avs_test_s_bs")
        os.makedirs(out_subdir, exist_ok=True)
        stale_jsonl = os.path.join(out_subdir, "inference_results.jsonl")
        with open(stale_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({"video_id": "stale_video"}) + "\n")

        args = DummyArgs(
            use_rl=False,
            stage="infer",
            test_name="test_s",
            benchmark="refavs",
            ckpt_dir=ckpt_dir,
            out_dir=out_dir,
            dry_run=False,
        )

        with patch("scripts.run_pipeline.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0)
            run_use_rl_false(args)

        with open(stale_jsonl, "r", encoding="utf-8") as f:
            data = json.loads(f.read().strip())
        assert data["video_id"] == "fresh_video", "Stale JSONL was not overwritten!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
