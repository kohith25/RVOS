from __future__ import annotations
#!/usr/bin/env python3
"""
scripts/rl_grpo/build_coldstart_data.py - Cold-Start Trajectory Generator for Action SFT

Generates canonical expert multi-turn trajectories from real training samples:
- Uses real tool calls: retrieve_evidence -> (inspect_frames) -> call_referdino -> call_sam2 -> stop.
- Selects candidate_id based on maximum IoU against GT masks (train-time oracle guidance only).
- Skips samples where best candidate IoU < 0.3.
- Null samples end with stop without call_sam2.
- Thought text varied across template pools (<= 60 words).
- Real observation texts recorded from tool server / mock client.
- Saves per-turn JSONL with prompt_text (build_turn_prompt), completion_text, and media paths.
"""

import os
import sys
import re
import json
import csv
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.constants import format_observation
from rl_grpo.rewards import compute_mask_iou
from tools.client import ToolClient, get_tool_client
from tools.mock_tools import MockToolServer


THOUGHT_TEMPLATES = {
    "retrieve_evidence": [
        "Query the multimodal knowledge graph to gather relational and acoustic context for '{expr}'.",
        "Retrieve acoustic evidence and spatial relationships from the knowledge graph for '{expr}'.",
        "Consult the knowledge graph for relational facts and audio cues regarding '{expr}'.",
        "Inspect the multimodal knowledge graph for sounding entities related to '{expr}'.",
        "Gather relational facts and acoustic events from the knowledge graph for '{expr}'.",
    ],
    "inspect_frames": [
        "Inspect frame range [0-9] to verify temporal acoustic alignment and visual presence.",
        "Examine video frames across seconds [0-9] to confirm audio activity and candidate positions.",
        "Check temporal frame intervals to align sounding events with visual detections.",
        "Verify visual and acoustic cues across the 10 frames before final candidate selection.",
        "Inspect video sequence [0-9] to correlate loudness peaks with moving objects.",
    ],
    "call_referdino": [
        "Generate candidate object tubelets for the primary target '{target}' using ReferDINO.",
        "Ground spatio-temporal tubelets matching '{target}' across the video clip.",
        "Identify visual candidate proposals corresponding to '{target}' with ReferDINO.",
        "Extract spatial candidates for referring expression '{target}' across all frames.",
        "Locate potential grounding candidates for '{target}' using clip-level ReferDINO.",
    ],
    "call_sam2": [
        "Candidate {cid} on the {side} exhibits high grounding score ({score:.2f}); segment with SAM2.",
        "Refine candidate {cid} located at {side} into precise pixel masks with SAM2.",
        "Segment candidate {cid} ({side}, grounding score {score:.2f}) across frames using SAM2.",
        "Apply SAM2 to candidate {cid} on the {side} to obtain frame-accurate segmentation masks.",
        "Generate pixel-level masks for candidate {cid} ({side}, score {score:.2f}) via SAM2.",
    ],
    "stop": [
        "Object successfully localized and segmented across all 10 frames; episode complete.",
        "Target tubelet fully segmented with high confidence; terminate reasoning episode.",
        "Audio-visual referring object successfully segmented; stop episode.",
        "Segmentation across the sequence is complete and verified; terminate episode.",
        "Reasoning goal reached with valid multi-frame masks; finish episode.",
    ],
    "stop_null": [
        "No matching sounding entity found in this video; terminate episode without segmentation.",
        "Null target detected in this clip; stop reasoning without generating masks.",
        "Reference refers to absent object; episode terminated correctly.",
        "Confirmed null sample; complete episode without SAM2 segmentation.",
        "Target object does not produce sound in this sequence; stop reasoning.",
    ]
}


def extract_objects_from_cot(cot_text: str) -> Tuple[Optional[str], Optional[str]]:
    f_obj = None
    s_obj = None
    if cot_text:
        f_match = re.search(r"<f_object>(.*?)</f_object>", cot_text, re.DOTALL)
        if f_match:
            f_obj = f_match.group(1).strip()
        s_match = re.search(r"<s_object>(.*?)</s_object>", cot_text, re.DOTALL)
        if s_match:
            s_obj = s_match.group(1).strip()
    return f_obj, s_obj


def load_gt_mask_sequence(mask_paths: List[str]) -> np.ndarray:
    """Load the ten ground-truth masks using the original dataset's mask layout."""
    missing = [path for path in mask_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Ground-truth mask file not found: {missing[0]}")
    masks = [np.asarray(Image.open(path).convert("L")) > 0 for path in mask_paths]
    return np.stack(masks, axis=0)


def select_candidate_by_mask_iou(
    candidates: List[Dict[str, Any]],
    gt_masks: np.ndarray,
    tool_client: Any,
    uid: str,
) -> Tuple[Optional[int], Dict[int, Dict[str, Any]]]:
    """Run SAM2 for each candidate and select only by the training GT mask IoU."""
    results = {}
    for candidate in candidates:
        candidate_id = int(candidate["cand_id"])
        boxes = candidate.get("boxes")
        if not boxes:
            raise ValueError(f"Candidate {candidate_id} has no boxes for SAM2.")
        masks, logits = tool_client.sam2_segment(uid, boxes)
        mask_array = np.asarray(masks, dtype=np.uint8)
        results[candidate_id] = {
            "masks": mask_array,
            "logits": np.asarray(logits, dtype=np.float32),
            "iou": compute_mask_iou(mask_array, gt_masks),
            "mean_mask_area": float((mask_array > 0).mean()),
        }
    if not results:
        return None, results
    best_candidate = max(results, key=lambda candidate_id: results[candidate_id]["iou"])
    return best_candidate, results


def build_expert_trajectory(
    expression: str,
    uid: str = "sample_0",
    mock: bool = False,
) -> List[Dict[str, Any]]:
    """Generate an explicitly mock-only scripted trajectory from mock tool results."""
    if not mock:
        raise ValueError("build_expert_trajectory is available only in --mock mode.")

    history = []
    turns = []
    mock_server = MockToolServer()
    mock_client = ToolClient(use_mock_fallback=True, use_cache=False)
    kg_result = mock_server.kg_retrieve(uid, expression)
    inspection = mock_client.inspect_frames(uid, 0, 9)
    candidates = mock_server.referdino_ground(uid, expression)
    if not candidates:
        raise RuntimeError(f"Mock ReferDINO returned no candidates for '{uid}'.")
    candidate = candidates[0]
    masks, logits = mock_server.sam2_segment(uid, candidate["boxes"])
    masks_array = np.asarray(masks, dtype=np.uint8)
    mask_area = float((masks_array > 0).mean())

    steps = [
        {
            "thought": f"Query the multimodal knowledge graph to gather relational and acoustic context for '{expression}'.",
            "action": "retrieve_evidence",
            "args": {"query": expression},
            "obs_data": kg_result,
            "tool_result": kg_result,
        },
        {
            "thought": "Inspect the full clip for observed audio activity and visible proposals.",
            "action": "inspect_frames",
            "args": {"start": 0, "end": 9},
            "obs_data": inspection,
            "tool_result": inspection,
        },
        {
            "thought": f"Ground candidate object tubelets matching '{expression}' using ReferDINO.",
            "action": "call_referdino",
            "args": {"text": expression},
            "obs_data": {"candidates": candidates},
            "tool_result": {"candidates": candidates},
        },
        {
            "thought": f"Refine candidate {candidate['cand_id']} bounding boxes into pixel-level masks using SAM2.",
            "action": "call_sam2",
            "args": {"candidate_id": candidate["cand_id"]},
            "obs_data": {
                "candidate_id": candidate["cand_id"],
                "num_frames": len(masks_array),
                "mean_mask_area": mask_area,
            },
            "tool_result": {
                "masks": masks,
                "logits": logits,
                "mean_mask_area": mask_area,
            },
        },
        {
            "thought": "Candidate segmented accurately across all 10 video frames; terminate reasoning episode.",
            "action": "stop",
            "args": {},
            "obs_data": {},
            "tool_result": {},
        }
    ]
    for step in steps:
        words = step["thought"].split()
        assert len(words) <= 60
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
            "observation": obs_text,
            "tool_result": step.get("tool_result", {}),
            "mock": True,
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
    parser.add_argument("--meta_csv_path", type=str, default="./R2AVSBench/RefAVSBench_metadata.csv")
    parser.add_argument("--cot_anno_path", type=str, default="./R2AVSBench/RefThinker_instruction_tuning_set.json")
    parser.add_argument("--media_dir", type=str, default="./REFAVS/media")
    parser.add_argument("--gt_mask_dir", type=str, default="./REFAVS/gt_mask")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--iou_threshold", type=float, default=0.30)
    parser.add_argument("--tool_server_url", type=str, default="http://127.0.0.1:8088")
    parser.add_argument("--mock", action="store_true", help="Generate synthetic coldstart dataset on CPU")
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    print("\n" + "=" * 65)
    print("  Building Cold-Start Action Format Trajectories")
    print(f"  Target Output: {args.out_path} | Max Samples: {args.num_samples} | Mock: {args.mock}")
    print("=" * 65)

    if args.mock:
        sample_exprs = [
            ("mock_video_person", "the person playing acoustic guitar on the left"),
            ("mock_video_dog", "the barking dog running across the grass"),
            ("mock_video_singer", "the woman singing into a microphone"),
            ("mock_video_piano", "the piano played on the right"),
            ("mock_video_violin", "the violin soloist performing in the orchestra"),
        ]
        with open(args.out_path, "w", encoding="utf-8") as out_f:
            for idx in range(args.num_samples):
                uid, expression = sample_exprs[idx % len(sample_exprs)]
                uid = f"{uid}_{idx:03d}"
                for turn in build_expert_trajectory(expression, uid=uid, mock=True):
                    out_f.write(json.dumps(turn) + "\n")
        print(f"[Cold-Start MOCK] Wrote {args.num_samples * 5} mock turns to {args.out_path}")
        return 0

    if not os.path.isfile(args.meta_csv_path):
        raise FileNotFoundError(f"Training metadata CSV not found: {args.meta_csv_path}")
    tool_client = get_tool_client(server_url=args.tool_server_url, use_mock_fallback=False)

    # 1. Load train samples
    samples = []
    cot_data = {}
    if os.path.exists(args.cot_anno_path):
        with open(args.cot_anno_path, "r", encoding="utf-8") as f:
            cot_data = json.load(f)

    with open(args.meta_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(samples) >= args.num_samples:
                break
            if len(row) < 5 or row[2] != "train":
                continue
            vid, uid, split, fid, exp = row[:5]
            uid_parts = uid.rsplit("_", 2)
            is_null = len(uid_parts) >= 2 and uid_parts[-2].lower() == "null"
            clean_vid = uid.rsplit("_", 2)[0]
            f_obj, s_obj = extract_objects_from_cot(
                cot_data.get(uid, {}).get("output", "")
            )
            samples.append({
                "uid": uid,
                "vid": clean_vid,
                "fid": fid,
                "expr": exp,
                "is_null": is_null,
                "f_object": f_obj,
                "s_object": s_obj,
                "video_path": os.path.join(args.media_dir, clean_vid, "frames"),
                "audio_path": os.path.join(args.media_dir, clean_vid, "audio.wav"),
                "mask_paths": [
                    os.path.join(args.gt_mask_dir, vid, f"fid_{fid}", f"0000{i}.png")
                    for i in range(10)
                ],
            })

    print(f"[Cold-Start] Processing {len(samples)} samples with real tool rollouts...")

    written_turns = 0
    skipped_low_iou = 0

    with open(args.out_path, "w", encoding="utf-8") as out_f:
        for idx, s in enumerate(samples):
            uid = s["uid"]
            expr = s["expr"]
            is_null = s["is_null"]
            f_obj = s["f_object"] or s["s_object"] or expr
            grounding_target = f_obj

            history = []
            sample_turns = []
            training_supervision = None

            # Step 1: retrieve_evidence
            t_idx = idx % len(THOUGHT_TEMPLATES["retrieve_evidence"])
            thought_kg = THOUGHT_TEMPLATES["retrieve_evidence"][t_idx].format(expr=expr)
            kg_res = tool_client.retrieve_evidence(uid, expr, k_hops=1)
            obs_kg = format_observation("retrieve_evidence", kg_res)

            sample_turns.append({
                "thought": thought_kg,
                "action": "retrieve_evidence",
                "args": {"query": expr},
                "observation": obs_kg,
                "tool_result": kg_res,
            })
            history.append({
                "thought": thought_kg,
                "action": "retrieve_evidence",
                "args": {"query": expr},
                "observation": obs_kg,
            })

            # Optional Step 1b: inspect_frames if temporal or audio cues in expr
            if any(cue in expr.lower() for cue in ["while", "when", "during", "sound", "loud", "play"]):
                t_insp = idx % len(THOUGHT_TEMPLATES["inspect_frames"])
                thought_insp = THOUGHT_TEMPLATES["inspect_frames"][t_insp]
                res_insp = tool_client.inspect_frames(
                    uid,
                    0,
                    9,
                    visible_candidate_ids=[],
                )
                obs_insp = format_observation("inspect_frames", res_insp)
                sample_turns.append({
                    "thought": thought_insp,
                    "action": "inspect_frames",
                    "args": {"start": 0, "end": 9},
                    "observation": obs_insp,
                    "tool_result": res_insp,
                })
                history.append({
                    "thought": thought_insp,
                    "action": "inspect_frames",
                    "args": {"start": 0, "end": 9},
                    "observation": obs_insp,
                })

            if is_null:
                # Null sample trajectory ends with stop (no call_sam2)
                t_stop = idx % len(THOUGHT_TEMPLATES["stop_null"])
                thought_stop = THOUGHT_TEMPLATES["stop_null"][t_stop]
                obs_stop = format_observation("stop", {})
                sample_turns.append({
                    "thought": thought_stop,
                    "action": "stop",
                    "args": {},
                    "observation": obs_stop,
                    "tool_result": {},
                })
            else:
                gt_masks = load_gt_mask_sequence(s["mask_paths"])
                # Step 2: call_referdino
                t_ref = idx % len(THOUGHT_TEMPLATES["call_referdino"])
                thought_ref = THOUGHT_TEMPLATES["call_referdino"][t_ref].format(target=grounding_target)
                cands = tool_client.referdino_ground(uid, grounding_target)
                obs_ref = format_observation("call_referdino", {"candidates": cands})
                if not cands:
                    skipped_low_iou += 1
                    continue

                sample_turns.append({
                    "thought": thought_ref,
                    "action": "call_referdino",
                    "args": {"text": grounding_target},
                    "observation": obs_ref,
                    "tool_result": {"candidates": cands},
                })
                history.append({
                    "thought": thought_ref,
                    "action": "call_referdino",
                    "args": {"text": grounding_target},
                    "observation": obs_ref,
                })

                best_cand_id, sam2_results = select_candidate_by_mask_iou(
                    cands,
                    gt_masks,
                    tool_client,
                    uid,
                )
                if best_cand_id is None:
                    skipped_low_iou += 1
                    continue
                best_iou = sam2_results[best_cand_id]["iou"]
                if best_iou < args.iou_threshold:
                    skipped_low_iou += 1
                    continue
                selected_cand = next(
                    candidate for candidate in cands
                    if int(candidate["cand_id"]) == best_cand_id
                )
                best_side = selected_cand.get("side", "center")
                best_score = selected_cand.get("score", 0.0)

                # Step 3: call_sam2
                t_sam = idx % len(THOUGHT_TEMPLATES["call_sam2"])
                thought_sam = THOUGHT_TEMPLATES["call_sam2"][t_sam].format(
                    cid=best_cand_id, side=best_side, score=best_score
                )
                selected_masks = sam2_results[best_cand_id]["masks"]
                sam_summary = {
                    "candidate_id": best_cand_id,
                    "num_frames": int(selected_masks.shape[0]),
                    "mean_mask_area": sam2_results[best_cand_id]["mean_mask_area"],
                }
                obs_sam = format_observation("call_sam2", sam_summary)

                sample_turns.append({
                    "thought": thought_sam,
                    "action": "call_sam2",
                    "args": {"candidate_id": best_cand_id},
                    "observation": obs_sam,
                    "tool_result": {
                        "masks": selected_masks.tolist(),
                        "logits": sam2_results[best_cand_id]["logits"].tolist(),
                        **sam_summary,
                    },
                })
                history.append({
                    "thought": thought_sam,
                    "action": "call_sam2",
                    "args": {"candidate_id": best_cand_id},
                    "observation": obs_sam,
                })

                # Step 4: stop
                t_stop = idx % len(THOUGHT_TEMPLATES["stop"])
                thought_stop = THOUGHT_TEMPLATES["stop"][t_stop]
                obs_stop = format_observation("stop", {})
                sample_turns.append({
                    "thought": thought_stop,
                    "action": "stop",
                    "args": {},
                    "observation": obs_stop,
                    "tool_result": {},
                })
                training_supervision = {
                    "candidate_id": best_cand_id,
                    "mask_iou": best_iou,
                }

            # Format turns for LoRA SFT: prompt_text, completion_text, media paths
            rolling_history = []
            for turn in sample_turns:
                words = turn["thought"].split()
                assert len(words) <= 60, f"Thought exceeds 60 words: {turn['thought']}"

                prompt = TGSModelAdapter.build_turn_prompt(expr, rolling_history)
                completion_dict = {
                    "thought": turn["thought"],
                    "action": turn["action"],
                    "args": turn["args"],
                }
                completion_text = json.dumps(completion_dict)

                out_record = {
                    "uid": uid,
                    "expression": expr,
                    "prompt_text": prompt,
                    "completion_text": completion_text,
                    "action": turn["action"],
                    "args": turn["args"],
                    "thought": turn["thought"],
                    "observation": turn["observation"],
                    "video_path": s["video_path"],
                    "audio_path": s["audio_path"],
                    "tool_result": turn.get("tool_result", {}),
                }
                if training_supervision is not None:
                    out_record["training_supervision"] = training_supervision
                out_f.write(json.dumps(out_record) + "\n")
                written_turns += 1

                rolling_history.append({
                    "thought": turn["thought"],
                    "action": turn["action"],
                    "args": turn["args"],
                    "observation": turn["observation"],
                })

    print(f"[Cold-Start SUCCESS] Wrote {written_turns} turns to {args.out_path}. Skipped low IoU: {skipped_low_iou}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
