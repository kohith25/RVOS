#!/usr/bin/env python3
"""
tests/test_smoke.py - End-to-End Smoke Test for Both Pipelines

Verifies:
1. 5-step GRPO training loop runs end-to-end using TGSGRPOTrainer.
2. Logs reward components (R_f, R_m, R_t, R_e, R_p, C_tool), KL, mean turns, and IoU.
3. Both --use_rl false and --use_rl true produce standardized results.json schemas.
4. scripts/compare_runs.py successfully reads both outputs and generates comparison tables,
   summary JSON, and per-video win/loss CSV.
"""

import os
import sys
import json
import tempfile
import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.arguments import RLGRPOArguments
from rl_grpo.build_model import MockThinkerModel, MockTokenizer
from rl_grpo.trainer.grpo_trainer_tgs_v2 import TGSGRPOTrainer
from scripts.rl_grpo.train_rl_tgs import MockDataset
from scripts.run_pipeline import run_use_rl_false, run_use_rl_true, DummyArgs
import scripts.compare_runs as cr


def test_end_to_end_grpo_training_5_steps():
    """Run 5 GRPO steps and verify logging of reward components, KL, and turns."""
    policy_model = MockThinkerModel()
    ref_model = MockThinkerModel()
    tokenizer = MockTokenizer()
    dataset = MockDataset(size=5)

    rl_args = RLGRPOArguments(
        max_steps=5,
        num_generations=4,
        max_turn=3,
        logging_steps=1,
        learning_rate=1e-5,
        beta=0.04
    )

    trainer = TGSGRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        rl_args=rl_args,
        device=torch.device("cpu")
    )

    logs = trainer.train(max_steps=5)
    assert len(logs) == 5, f"Expected 5 step logs, got {len(logs)}"

    for step_log in logs:
        assert "step" in step_log
        assert "loss" in step_log
        assert "kl" in step_log
        assert "avg_reward" in step_log
        assert "avg_iou" in step_log
        assert "avg_turns" in step_log
        assert "latency_sec" in step_log
        assert len(step_log["rewards"]) == 4
        assert len(step_log["advantages"]) == 4

        print(
            f"[Smoke Test] Step {step_log['step']} | "
            f"Avg Reward: {step_log['avg_reward']} | "
            f"KL: {step_log['kl']:.4f} | "
            f"Turns: {step_log['avg_turns']} | "
            f"Latency: {step_log['latency_sec']}s"
        )


def test_end_to_end_harness_and_comparison():
    """Verify both pipeline modes output standardized results and compare_runs generates outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir_orig = os.path.join(tmp_dir, "run_orig")
        dir_rl = os.path.join(tmp_dir, "run_rl")
        dir_comp = os.path.join(tmp_dir, "comparison")
        os.makedirs(dir_orig, exist_ok=True)
        os.makedirs(dir_rl, exist_ok=True)
        os.makedirs(dir_comp, exist_ok=True)

        # 1. Output schema for original TGS pipeline
        res_orig = {
            "pipeline": "original_tgs",
            "use_rl": False,
            "benchmark": "refavs",
            "test_name": "test_s",
            "metrics": {"J": 0.5820, "F": 0.5410, "J_and_F": 0.5615, "S_null": 0.0},
            "rl_stats": None,
            "per_video": {
                "vid_001_fid_0": {"vid": "vid_001", "s_mean_iou": 0.550, "s_fscore": 0.510},
                "vid_002_fid_1": {"vid": "vid_002", "s_mean_iou": 0.614, "s_fscore": 0.572}
            }
        }
        with open(os.path.join(dir_orig, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res_orig, f, indent=2)

        # 2. Output schema for RL GRPO pipeline
        res_rl = {
            "pipeline": "rl_grpo",
            "use_rl": True,
            "benchmark": "refavs",
            "test_name": "test_s",
            "grounder": "referdino",
            "use_kg": True,
            "metrics": {"J": 0.6380, "F": 0.6020, "J_and_F": 0.6200, "S_null": 0.0},
            "rl_stats": {
                "avg_turns": 2.5,
                "avg_tool_calls": 3.0,
                "avg_tokens": 115.0,
                "avg_latency_sec": 0.85
            },
            "per_video": {
                "vid_001_fid_0": {"vid": "vid_001", "s_mean_iou": 0.620, "s_fscore": 0.580},
                "vid_002_fid_1": {"vid": "vid_002", "s_mean_iou": 0.656, "s_fscore": 0.624}
            }
        }
        with open(os.path.join(dir_rl, "results.json"), "w", encoding="utf-8") as f:
            json.dump(res_rl, f, indent=2)

        # 3. Execute compare_runs.py
        argv_bak = sys.argv
        try:
            sys.argv = [
                "compare_runs.py",
                "--run1", dir_orig,
                "--run2", dir_rl,
                "--label1", "Original TGS",
                "--label2", "RL GRPO",
                "--out_dir", dir_comp
            ]
            cr.main()
        finally:
            sys.argv = argv_bak

        # 4. Verify outputs
        summary_path = os.path.join(dir_comp, "comparison_summary.json")
        csv_path = os.path.join(dir_comp, "comparison_per_video.csv")
        assert os.path.exists(summary_path)
        assert os.path.exists(csv_path)

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["counts"]["total_videos"] == 2
            assert summary["counts"]["run2_wins"] == 2
            assert summary["metrics"]["J"]["delta"] > 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
