#!/usr/bin/env python3
"""
rl_grpo/constants.py - Action Space, Schemas, Costs, and Prompts
"""

import json
import re
from typing import Tuple, Optional, Dict, Any

ALLOWED_ACTIONS = [
    "retrieve_evidence",
    "inspect_frames",
    "refine_description",
    "call_referdino",
    "call_sam2",
    "stop",
]

# Action costs (C_tool penalty)
ACTION_TO_COST = {
    "retrieve_evidence": 0.15,
    "inspect_frames": 0.15,
    "refine_description": 0.10,
    "call_referdino": 0.20,
    "call_sam2": 0.25,
    "stop": 0.0,
}

SYSTEM_INSTRUCTION = (
    "You are Ref-Thinker, an audio-visual reasoning agent for video referring object segmentation.\n"
    "You perceive video frames, audio signals, and a natural language referring expression.\n"
    "At each step, reason about the audio-visual evidence and decide the best next action.\n"
    "Your output MUST contain a single valid JSON object with the following schema:\n"
    "{\n"
    '  "thought": "<brief step-by-step reasoning>",\n'
    '  "action": "<action_name>",\n'
    '  "args": { ... }\n'
    "}\n\n"
    "Available actions and required arguments:\n"
    '1. retrieve_evidence: {"query": "<text description of object, spatial relation, or sound>"}\n'
    '2. inspect_frames: {"start": <frame_index_0_to_9>, "end": <frame_index_0_to_9>}\n'
    '3. refine_description: {"text": "<refined object description focusing on visual & audio cues>"}\n'
    '4. call_referdino: {"text": "<grounding expression for candidate generation>"}\n'
    '5. call_sam2: {"candidate_id": <int_id_from_referdino_candidates>}\n'
    '6. stop: {}\n'
    "Note: Always ground the object with call_referdino and segment with call_sam2 before stopping."
)


def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extracts a JSON object from text, handling markdown code blocks, think tags, and surrounding prose.
    """
    if not text:
        return None

    # Strip any <think> ... </think> or keep it inside
    clean_text = text.strip()

    # If wrapped in markdown ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Search for first outermost balanced curly braces {...}
    start = clean_text.find("{")
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(clean_text)):
            if clean_text[i] == "{":
                depth += 1
            elif clean_text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end != -1:
            try:
                return json.loads(clean_text[start:end])
            except json.JSONDecodeError:
                pass

    return None


def parse_and_validate_action(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Parses and strictly validates the model's action output.
    Returns:
        (action_dict, error_message)
        If valid: (action_dict, None)
        If invalid: (None, "Reason for validation failure")
    """
    data = extract_json_from_text(text)
    if data is None or not isinstance(data, dict):
        return None, "Failed to parse a valid JSON object from model response."

    action = data.get("action")
    if not action or not isinstance(action, str):
        return None, "Missing or invalid 'action' key in JSON."

    action = action.strip()
    if action not in ALLOWED_ACTIONS:
        return None, f"Action '{action}' is not in allowed actions: {ALLOWED_ACTIONS}."

    args = data.get("args", {})
    if not isinstance(args, dict):
        return None, f"'args' must be a dictionary, got {type(args).__name__}."

    # Validate specific action argument constraints
    if action == "retrieve_evidence":
        if "query" not in args or not isinstance(args["query"], str) or not args["query"].strip():
            return None, "Action 'retrieve_evidence' requires non-empty string argument 'query'."

    elif action == "inspect_frames":
        if "start" not in args or "end" not in args:
            return None, "Action 'inspect_frames' requires integer arguments 'start' and 'end'."
        try:
            start, end = int(args["start"]), int(args["end"])
            if start < 0 or end > 9 or start > end:
                return None, f"Frame range [{start}, {end}] must be within [0, 9] with start <= end."
        except (ValueError, TypeError):
            return None, "Arguments 'start' and 'end' must be integers."

    elif action == "refine_description":
        if "text" not in args or not isinstance(args["text"], str) or not args["text"].strip():
            return None, "Action 'refine_description' requires non-empty string argument 'text'."

    elif action == "call_referdino":
        if "text" not in args or not isinstance(args["text"], str) or not args["text"].strip():
            return None, "Action 'call_referdino' requires non-empty string argument 'text'."

    elif action == "call_sam2":
        if "candidate_id" not in args:
            return None, "Action 'call_sam2' requires integer argument 'candidate_id'."
        try:
            cand_id = int(args["candidate_id"])
            if cand_id < 0:
                return None, "Argument 'candidate_id' must be non-negative."
        except (ValueError, TypeError):
            return None, "Argument 'candidate_id' must be an integer."

    return data, None


def format_observation(action: str, result: Any) -> str:
    """
    Renders tool execution results into natural-language observation text for next agent turn.
    """
    if action == "retrieve_evidence":
        facts = result.get("facts", [])
        if facts:
            facts_str = "\n".join(f"- {f}" for f in facts)
            return f"Retrieved Knowledge Graph Evidence:\n{facts_str}"
        return "Retrieved Knowledge Graph Evidence: No relevant relational facts found."

    elif action == "inspect_frames":
        start = result.get("start", 0)
        end = result.get("end", 9)
        loudness = result.get("avg_audio_loudness", 0.0)
        cands = result.get("visible_candidate_ids", [])
        active_secs = result.get("active_audio_secs", list(range(start, end + 1)))
        return f"Inspected video frames [{start}-{end}]. Avg audio loudness: {loudness:.2f}, active frames: {active_secs}, visible candidates: {cands}."

    elif action == "refine_description":
        refined = result.get("refined_text", "")
        return f"Description refined to: '{refined}'."

    elif action == "call_referdino":
        candidates = result.get("candidates", [])
        if not candidates:
            return "ReferDINO Grounding: No candidate objects found matching the description."
        lines = ["ReferDINO generated candidates:"]
        for c in candidates:
            cid = c.get("cand_id", 0)
            side = c.get("side", "center")
            score = c.get("score", 0.0)
            area_pct = int(c.get("mean_area", 0.0) * 100)
            lines.append(f"- Candidate {cid}: location={side}, grounding_score={score:.2f}, mean_area={area_pct}%")
        return "\n".join(lines)

    elif action == "call_sam2":
        cand_id = result.get("candidate_id", 0)
        num_frames = result.get("num_frames", 10)
        mean_mask_area = result.get("mean_mask_area", 0.0)
        return (f"SAM2 Segmentation: Generated pixel masks for candidate {cand_id} across {num_frames} frames "
                f"(average mask coverage: {mean_mask_area * 100:.1f}%).")

    elif action == "stop":
        return "Episode terminated by agent."

    return f"Observation: {str(result)}"
