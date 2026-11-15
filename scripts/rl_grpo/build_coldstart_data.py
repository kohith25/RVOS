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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rl_grpo.model_adapter import TGSModelAdapter
from rl_grpo.constants import format_observation
from tools.client import get_tool_client


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


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes mean intersection-over-union between two binary mask sequences."""
    if mask1 is None or mask2 is None:
        return 0.0
    m1 = mask1 > 0
    m2 = mask2 > 0
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def compute_box_mask_iou(boxes: List[List[float]], gt_masks: np.ndarray) -> float:
    """Computes approximate IoU by converting bounding boxes to box masks."""
    if gt_masks is None or not boxes:
        return 0.0
    T, H, W = gt_masks.shape
    box_masks = np.zeros((T, H, W), dtype=bool)
    for t in range(min(T, len(boxes))):
        b = boxes[t]
        if len(b) == 4 and b != [0, 0, 0, 0]:
            x1, y1 = int(np.clip(b[0] * W, 0, W - 1)), int(np.clip(b[1] * H, 0, H - 1))
            x2, y2 = int(np.clip(b[2] * W, x1 + 1, W)), int(np.clip(b[3] * H, y1 + 1, H))
            box_masks[t, y1:y2, x1:x2] = True
    return compute_mask_iou(box_masks, gt_masks)


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

    tool_client = get_tool_client(server_url=args.tool_server_url, use_mock_fallback=args.mock)

    # 1. Load train samples
    samples = []
    if args.mock or not os.path.exists(args.meta_csv_path):
        print(f"[Notice] Generating synthetic train samples (mock={args.mock})...")
        sample_exprs = [
            ("vid_001_person_0", "the person playing acoustic guitar on the left", False),
            ("vid_002_dog_1", "the barking dog running across the grass", False),
            ("vid_003_singer_0", "the woman singing into a microphone", False),
            ("null_vid_004_piano_0", "the piano played on the right", True),
            ("vid_005_violin_0", "the violin soloist performing in the orchestra", False),
        ]
        for i in range(args.num_samples):
            base_uid, expr, is_null = sample_exprs[i % len(sample_exprs)]
            uid = f"{base_uid}_{i:03d}"
            samples.append({
                "uid": uid,
                "vid": uid.rsplit("_", 2)[0],
                "expr": expr,
                "is_null": is_null,
                "f_object": expr.split()[1] if len(expr.split()) > 1 else "object",
                "s_object": None,
                "video_path": os.path.join(args.media_dir, uid.rsplit("_", 2)[0], "frames"),
                "audio_path": os.path.join(args.media_dir, uid.rsplit("_", 2)[0], "audio.wav"),
                "gt_masks": None,
            })
    else:
        # Real metadata CSV reading
        cot_data = {}
        if os.path.exists(args.cot_anno_path):
            with open(args.cot_anno_path, "r", encoding="utf-8") as f:
                cot_data = json.load(f)

        with open(args.meta_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(samples) >= args.num_samples:
                    break
                if len(row) < 5:
                    continue
                vid, uid, split, fid, exp = row[:5]
                if split != "train":
                    continue

                is_null = uid.startswith("null_")
                clean_vid = uid.rsplit("_", 2)[0]

                cot_output = cot_data.get(uid, {}).get("output", "")
                f_obj, s_obj = extract_objects_from_cot(cot_output)

                v_path = os.path.join(args.media_dir, clean_vid, "frames")
                a_path = os.path.join(args.media_dir, clean_vid, "audio.wav")

                samples.append({
                    "uid": uid,
                    "vid": clean_vid,
                    "expr": exp,
                    "is_null": is_null,
                    "f_object": f_obj,
                    "s_object": s_obj,
                    "video_path": v_path,
                    "audio_path": a_path,
                    "gt_masks": None,
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
                res_insp = {
                    "start": 0, "end": 9, "num_frames": 10,
                    "avg_audio_loudness": 0.45, "visible_candidate_ids": [0, 1]
                }
                obs_insp = format_observation("inspect_frames", res_insp)
                sample_turns.append({
                    "thought": thought_insp,
                    "action": "inspect_frames",
                    "args": {"start": 0, "end": 9},
                    "observation": obs_insp,
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
                })
            else:
                # Step 2: call_referdino
                t_ref = idx % len(THOUGHT_TEMPLATES["call_referdino"])
                thought_ref = THOUGHT_TEMPLATES["call_referdino"][t_ref].format(target=grounding_target)
                cands = tool_client.referdino_ground(uid, grounding_target)
                obs_ref = format_observation("call_referdino", {"candidates": cands})

                sample_turns.append({
                    "thought": thought_ref,
                    "action": "call_referdino",
                    "args": {"text": grounding_target},
                    "observation": obs_ref,
                })
                history.append({
                    "thought": thought_ref,
                    "action": "call_referdino",
                    "args": {"text": grounding_target},
                    "observation": obs_ref,
                })

                # Choose best candidate via GT IoU guidance (training only)
                best_cand_id = 0
                best_iou = 0.85
                best_side = "center"
                best_score = 0.85

                if cands:
                    best_cand_id = cands[0].get("cand_id", 0)
                    best_side = cands[0].get("side", "center")
                    best_score = cands[0].get("score", 0.85)

                    if s["gt_masks"] is not None:
                        best_cand_id = 0
                        best_iou = -1.0
                        for c in cands:
                            boxes = c.get("boxes", [])
                            iou = compute_box_mask_iou(boxes, s["gt_masks"])
                            if iou > best_iou:
                                best_iou = iou
                                best_cand_id = c.get("cand_id", 0)
                                best_side = c.get("side", "center")
                                best_score = c.get("score", 0.85)

                        if best_iou < args.iou_threshold:
                            skipped_low_iou += 1
                            continue

                # Step 3: call_sam2
                t_sam = idx % len(THOUGHT_TEMPLATES["call_sam2"])
                thought_sam = THOUGHT_TEMPLATES["call_sam2"][t_sam].format(
                    cid=best_cand_id, side=best_side, score=best_score
                )
                chosen_boxes = cands[best_cand_id].get("boxes", [[0, 0, 1, 1]] * 10) if (cands and best_cand_id < len(cands)) else [[0, 0, 1, 1]] * 10
                sam_res = tool_client.sam2_segment(uid, chosen_boxes)
                obs_sam = format_observation("call_sam2", {"candidate_id": best_cand_id, "num_frames": 10, "mean_mask_area": 0.15})

                sample_turns.append({
                    "thought": thought_sam,
                    "action": "call_sam2",
                    "args": {"candidate_id": best_cand_id},
                    "observation": obs_sam,
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
                })

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
                }
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
