from __future__ import annotations
#!/usr/bin/env python3
"""
scripts/rl_grpo/build_coldstart_data.py - Cold-Start Trajectory Generator for Action SFT

Generates canonical expert multi-turn trajectories:
Turn 0: retrieve_evidence
Turn 1: call_referdino
Turn 2: call_sam2
Turn 3: stop

All thoughts are strictly <= 60 words.
Formatted with build_turn_prompt for LoRA SFT token masking.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.constants import format_observation


def build_expert_trajectory(expression: str, uid: str = "sample_0") -> List[Dict[str, Any]]:
    """Generates 4 scripted turns with build_turn_prompt and formatted completion JSON."""
    history = []
    turns = []

    steps = [
        {
            "thought": f"Retrieve relational facts and sounding entities from knowledge graph for '{expression}'.",
            "action": "retrieve_evidence",
            "args": {"query": expression},
            "obs_data": {"facts": [f"Visual entity '{expression}' localized in video frames", "Acoustic event active across sequence"]}
        },
        {
            "thought": f"Ground candidate object tubelets matching '{expression}' using ReferDINO.",
            "action": "call_referdino",
            "args": {"text": expression},
            "obs_data": {"candidates": [{"cand_id": 0, "side": "center", "score": 0.88, "mean_area": 0.14}]}
        },
        {
            "thought": "Refine the grounded candidate 0 bounding boxes into pixel-accurate binary masks using SAM2.",
            "action": "call_sam2",
            "args": {"candidate_id": 0},
            "obs_data": {"candidate_id": 0, "num_frames": 10, "mean_mask_area": 0.14}
        },
        {
            "thought": "Candidate segmented accurately across all 10 video frames; terminate reasoning episode.",
            "action": "stop",
            "args": {},
            "obs_data": {}
        }
    ]

    for step in steps:
        # Check word count constraint (<= 60 words)
        words = step["thought"].split()
        assert len(words) <= 60, f"Thought exceeds 60 words: {step['thought']}"

        prompt = TGSModelAdapter.build_turn_prompt(expression, history)
        completion_dict = {
            "thought": step["thought"],
            "action": step["action"],
            "args": step["args"]
        }
        completion_text = json.dumps(completion_dict)

        obs_text = format_observation(step["action"], step["obs_data"])

        turns.append({
            "uid": uid,
            "expression": expression,
            "prompt_text": prompt,
            "completion_text": completion_text,
            "action": step["action"],
            "args": step["args"],
            "thought": step["thought"],
            "observation": obs_text
        })

        history.append({
            "thought": step["thought"],
            "action": step["action"],
            "args": step["args"],
            "observation": obs_text
        })

    return turns


def get_parser():
    parser = argparse.ArgumentParser(description="Build Cold-Start SFT Data for Ref-Thinker Action Format")
    parser.add_argument("--out_path", type=str, default="data/coldstart_trajectories.jsonl")
    parser.add_argument("--meta_csv_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--mock", action="store_true", help="Generate synthetic coldstart dataset")
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args)


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    samples = []
    if args.mock or not args.meta_csv_path or not os.path.exists(args.meta_csv_path or ""):
        # Synthetic samples for testing
        print(f"[Cold-Start] Generating {args.num_samples} synthetic expert trajectories...")
        expressions = [
            "the man playing acoustic guitar on the left",
            "the barking dog running across the grass",
            "the woman singing into a microphone",
            "the piano played by a child",
            "the violin soloist in the orchestra",
        ]
        for i in range(args.num_samples):
            expr = expressions[i % len(expressions)]
            samples.append((f"sample_{i:04d}", expr))
    else:
        import pandas as pd
        df = pd.read_csv(args.meta_csv_path)
        print(f"[Cold-Start] Loaded {len(df)} metadata rows from {args.meta_csv_path}")
        count = 0
        for _, row in df.iterrows():
            if count >= args.num_samples:
                break
            uid = str(row.get("uid", f"sample_{count}"))
            expr = str(row.get("ref", row.get("expression", "")))
            if expr and expr != "null":
                samples.append((uid, expr))
                count += 1

    total_turns = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for uid, expr in samples:
            traj_turns = build_expert_trajectory(expr, uid=uid)
            for turn in traj_turns:
                f.write(json.dumps(turn) + "\n")
                total_turns += 1

    print(f"[Cold-Start] Successfully wrote {total_turns} expert turns to {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
