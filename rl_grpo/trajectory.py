from __future__ import annotations
#!/usr/bin/env python3
"""
rl_grpo/trajectory.py - Multi-turn Trajectory Rollout Manager

Resolves P2 & P5:
1. Replaces RNG mock boxes and GT oracle leaks with real tool calls via ToolClient.
2. Tools never receive Ground Truth annotations; candidate selection is made solely by Ref-Thinker.
3. Records per-turn token sequences (prompt_ids, completion_ids, old_logps) for loss calculation.
4. Manages multi-turn state, observations, and episodic termination.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from rl_grpo.constants import (
    ACTION_TO_COST,
    format_observation,
    get_allowed_actions,
    get_system_instruction,
)
from rl_grpo.model_adapter import TGSModelAdapter, ModelAction
from rl_grpo.rewards import compute_episodic_reward, compute_mask_iou
from tools.client import ToolClient, get_tool_client


@dataclass
class TurnRecord:
    turn: int
    action: str
    args: Dict[str, Any]
    thought: str
    raw_text: str
    observation: str
    is_valid: bool
    prompt_ids: List[int]
    completion_ids: List[int]
    old_logps: List[float]
    pred_masks: Optional[np.ndarray] = None
    pred_logits: Optional[np.ndarray] = None
    tool_cost: float = 0.0


@dataclass
class Trajectory:
    uid: str
    expression: str
    turns: List[TurnRecord]
    final_pred_masks: Optional[np.ndarray]
    final_pred_logits: Optional[np.ndarray] = None
    sam2_ious: List[float] = field(default_factory=list)
    reward_dict: Dict[str, float] = field(default_factory=dict)
    video_tensor: Optional[Any] = None
    audio_tensor: Optional[Any] = None

    @property
    def total_reward(self) -> float:
        return self.reward_dict.get("reward", 0.0)


class TrajectoryCollector:
    def __init__(
        self,
        model_adapter: TGSModelAdapter,
        tool_client: Optional[ToolClient] = None,
        max_turns: int = 4,
        use_kg: bool = True,
    ):
        self.model_adapter = model_adapter
        self.tool_client = tool_client if tool_client is not None else get_tool_client()
        self.max_turns = max_turns
        self.use_kg = use_kg

    def rollout(
        self,
        sample: Dict[str, Any],
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> Trajectory:
        """
        Executes a complete multi-turn episode rollout for a sample.
        """
        uid = sample.get("uid", "sample_0")
        expression = sample.get("expression") or sample.get("ref", "")
        video_tensor = sample.get("video")
        audio_tensor = sample.get("audio")
        gt_masks = sample.get("gt_masks")
        is_null = sample.get("is_null", False)

        history: List[Dict[str, Any]] = []
        turns: List[TurnRecord] = []

        current_candidates: List[Dict[str, Any]] = []
        final_pred_masks: Optional[np.ndarray] = None
        final_pred_logits: Optional[np.ndarray] = None
        sam2_ious: List[float] = []

        for turn_idx in range(self.max_turns):
            # 1. Policy samples next action
            model_action: ModelAction = self.model_adapter.sample_action(
                expression=expression,
                video_tensor=video_tensor,
                audio_tensor=audio_tensor,
                history=history,
                turn_index=turn_idx,
                temperature=temperature,
                top_p=top_p
            )

            action = model_action.action_name
            args = model_action.args
            is_valid = model_action.is_valid
            cost = ACTION_TO_COST.get(action, 0.05) if is_valid else 0.10

            turn_pred_masks = None
            observation_text = ""

            # 2. Tool Execution (WITHOUT any GT leak)
            if not is_valid:
                observation_text = f"Action Error: {model_action.error}. Please output valid JSON."

            elif action == "retrieve_evidence":
                if not self.use_kg:
                    is_valid = False
                    cost = 0.50
                    observation_text = "Action Error: retrieve_evidence is disabled when --use_kg false. Do not call KG retrieval."
                else:
                    query = args.get("query", expression)
                    res = self.tool_client.retrieve_evidence(uid, query, k_hops=1)
                    observation_text = format_observation("retrieve_evidence", res)

            elif action == "inspect_frames":
                start = args.get("start", 0)
                end = args.get("end", 9)
                if self.use_kg:
                    kg_info = self.tool_client.kg_retrieve(uid, query="", k_hops=1)
                    audio_nodes = [n for n in kg_info.get("nodes", {}).values() if n.get("type") == "audio_event"]
                    range_loudness = [n.get("loudness", 0.0) for n in audio_nodes if start <= n.get("interval", [0])[0] <= end]
                    avg_loudness = round(sum(range_loudness) / len(range_loudness), 3) if range_loudness else 0.35
                else:
                    if audio_tensor is not None and hasattr(audio_tensor, "abs"):
                        avg_loudness = round(float(audio_tensor.abs().mean().item()) * 10, 3)
                    else:
                        avg_loudness = 0.35

                visible_cands = [c.get("cand_id") for c in current_candidates]
                res = {
                    "start": start,
                    "end": end,
                    "num_frames": end - start + 1,
                    "avg_audio_loudness": avg_loudness,
                    "visible_candidate_ids": visible_cands,
                    "resolution": [224, 224]
                }
                observation_text = format_observation("inspect_frames", res)

            elif action == "refine_description":
                text = args.get("text", expression)
                res = {"refined_text": text}
                observation_text = format_observation("refine_description", res)

            elif action == "call_referdino":
                text = args.get("text", expression)
                current_candidates = self.tool_client.referdino_ground(uid, text)
                res = {"candidates": current_candidates}
                observation_text = format_observation("call_referdino", res)

            elif action == "call_sam2":
                cand_id = args.get("candidate_id", 0)
                # FIX 7(c): If ReferDINO was never called or candidate_id is unknown, invalid action penalty (-0.5)
                # Do NOT fabricate default boxes!
                chosen_boxes = None
                if current_candidates:
                    for c in current_candidates:
                        if c.get("cand_id") == cand_id:
                            chosen_boxes = c.get("boxes")
                            break

                if chosen_boxes is None:
                    is_valid = False
                    cost = 0.50  # Penalty for unknown candidate
                    observation_text = f"Action Error: candidate_id {cand_id} was not produced by ReferDINO (available candidates: {[c.get('cand_id') for c in current_candidates]}). Call call_referdino first."
                    turn_pred_masks = None
                else:
                    raw_masks, raw_logits = self.tool_client.sam2_segment(uid, chosen_boxes)
                    turn_pred_masks = np.array(raw_masks, dtype=np.uint8) if isinstance(raw_masks, list) else raw_masks
                    turn_pred_logits = np.array(raw_logits, dtype=np.float32) if isinstance(raw_logits, list) else raw_logits
                    final_pred_masks = turn_pred_masks
                    final_pred_logits = turn_pred_logits

                    # Compute SAM2 IoU for progressive reward tracking
                    if gt_masks is not None and not is_null and turn_pred_masks is not None:
                        cur_iou = compute_mask_iou(turn_pred_masks, gt_masks)
                        sam2_ious.append(cur_iou)

                    coverage = float((turn_pred_masks > 0).sum() / turn_pred_masks.size) if turn_pred_masks is not None else 0.0
                    res = {"candidate_id": cand_id, "num_frames": 10, "mean_mask_area": round(coverage, 3)}
                    observation_text = format_observation("call_sam2", res)

            elif action == "stop":
                observation_text = format_observation("stop", {})

            # 3. Record turn
            turn_record = TurnRecord(
                turn=turn_idx,
                action=action,
                args=args,
                thought=model_action.thought,
                raw_text=model_action.raw_text,
                observation=observation_text,
                is_valid=is_valid,
                prompt_ids=model_action.prompt_ids,
                completion_ids=model_action.completion_ids,
                old_logps=model_action.old_logps,
                pred_masks=turn_pred_masks,
                pred_logits=turn_pred_logits if action == "call_sam2" else None,
                tool_cost=cost
            )
            turns.append(turn_record)

            history.append({
                "thought": model_action.thought,
                "action": action,
                "args": args,
                "observation": observation_text
            })

            # Termination condition
            if action == "stop":
                break

        # 4. Compute Episodic Reward at end of trajectory
        turn_dicts = [{"action": t.action, "args": t.args, "is_valid": t.is_valid} for t in turns]
        reward_dict = compute_episodic_reward(
            turns=turn_dicts,
            final_pred_masks=final_pred_masks,
            gt_masks=gt_masks,
            expression=expression,
            is_null=is_null,
            sam2_ious=sam2_ious
        )

        return Trajectory(
            uid=uid,
            expression=expression,
            turns=turns,
            final_pred_masks=final_pred_masks,
            final_pred_logits=final_pred_logits,
            sam2_ious=sam2_ious,
            reward_dict=reward_dict,
            video_tensor=video_tensor,
            audio_tensor=audio_tensor,
        )
