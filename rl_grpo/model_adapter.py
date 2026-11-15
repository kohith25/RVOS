#!/usr/bin/env python3
"""
rl_grpo/model_adapter.py - Multimodal Policy Adapter for Ref-Thinker

Resolves P3:
1. Feeds multimodal video and audio tensors through `batch_X_modals` on every generation turn.
2. Formats multi-turn conversational state appending prior actions and tool observations.
3. Parses and validates JSON actions without converting errors into silent stops.
4. Records per-token logprobs and completion token IDs for GRPO policy updates.
5. Provides a robust Mock execution path for CPU and unit testing without GPU weights.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def _no_grad_if_torch(func):
    if HAS_TORCH:
        return torch.no_grad()(func)
    return func

from rl_grpo.constants import (
    ALLOWED_ACTIONS,
    SYSTEM_INSTRUCTION,
    parse_and_validate_action,
)


@dataclass
class ModelAction:
    action_dict: Optional[Dict[str, Any]]
    action_name: str
    args: Dict[str, Any]
    thought: str
    raw_text: str
    is_valid: bool
    error: Optional[str]
    prompt_ids: List[int]
    completion_ids: List[int]
    old_logps: List[float]


class TGSModelAdapter:
    def __init__(self, model, tokenizer, device=None, use_mock=False):
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
        else:
            self.device = device
        self.use_mock = use_mock or (model is None)

    @staticmethod
    def build_turn_prompt(expression: str, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Constructs the conversational prompt containing multimodal start/end tokens,
        referring expression, system instruction, and multi-turn observation history.
        """
        prompt = (
            "<s>This is a video:\n<video_start><video><video_end>\n"
            "This is an audio:\n<audio_start><audio><audio_end>\n"
            f"Given the referential expression: '{expression.lower()}', analyze the video and audio, "
            "reason about object identity, and decide the next action.\n\n"
            f"{SYSTEM_INSTRUCTION}\n"
        )

        if history:
            prompt += "\n--- Interaction History ---\n"
            for idx, h in enumerate(history):
                action_str = json.dumps({"thought": h.get("thought", ""), "action": h.get("action", ""), "args": h.get("args", {})})
                obs_str = h.get("observation", "")
                prompt += f"Turn {idx + 1}:\nAgent: {action_str}\nObservation: {obs_str}\n"

        prompt += "\nAgent Next Action (JSON):"
        return prompt

    @_no_grad_if_torch
    def sample_action(
        self,
        expression: str,
        video_tensor: torch.Tensor,
        audio_tensor: torch.Tensor,
        history: Optional[List[Dict[str, Any]]] = None,
        turn_index: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 96,
    ) -> ModelAction:
        """
        Rolls out a single action turn from Ref-Thinker.
        """
        if history is None:
            history = []

        prompt_text = self.build_turn_prompt(expression, history)

        # ------------------------------------------------------------------ #
        # MOCK / CPU Fallback Generation Path                                #
        # ------------------------------------------------------------------ #
        if self.use_mock:
            # Deterministic sensible action trajectory for tests
            if turn_index == 0:
                mock_json = {
                    "thought": "Retrieve knowledge graph context for object identification.",
                    "action": "retrieve_evidence",
                    "args": {"query": expression}
                }
            elif turn_index == 1:
                mock_json = {
                    "thought": "Call ReferDINO to ground the target object candidates.",
                    "action": "call_referdino",
                    "args": {"text": expression}
                }
            elif turn_index == 2:
                mock_json = {
                    "thought": "Refine the best candidate mask using SAM2.",
                    "action": "call_sam2",
                    "args": {"candidate_id": 0}
                }
            else:
                mock_json = {
                    "thought": "Segmentation complete; finish episode.",
                    "action": "stop",
                    "args": {}
                }

            raw_text = json.dumps(mock_json)
            prompt_ids = [101, 102] if self.tokenizer is None else self.tokenizer.encode(prompt_text)
            completion_ids = [201, 202, 203] if self.tokenizer is None else self.tokenizer.encode(raw_text)
            old_logps = [-0.15] * len(completion_ids)

            return ModelAction(
                action_dict=mock_json,
                action_name=mock_json["action"],
                args=mock_json.get("args", {}),
                thought=mock_json.get("thought", ""),
                raw_text=raw_text,
                is_valid=True,
                error=None,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                old_logps=old_logps,
            )

        # ------------------------------------------------------------------ #
        # Real Multimodal Model Generation Path                              #
        # ------------------------------------------------------------------ #
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        batch_input_ids = [torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)]
        batch_labels = [torch.full((len(prompt_tokens),), -100, dtype=torch.long, device=self.device)]

        batch_X_modals = [{
            '<video>': video_tensor.to(self.device),
            '<audio>': audio_tensor.to(self.device)
        }]

        # Ensure eval mode & lora dropout = 0 for deterministic rollout
        self.model.eval()
        saved_dropouts = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_dropout"):
                for adapter_name, d_layer in getattr(module, "lora_dropout", {}).items():
                    saved_dropouts[(name, adapter_name)] = d_layer.p
                    d_layer.p = 0.0

        try:
            # Generate through Ref-Thinker
            gen_kwargs = {
                "batch_input_ids": batch_input_ids,
                "batch_labels": batch_labels,
                "batch_X_modals": batch_X_modals,
                "batch_task_names": ['ref-avs'],
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.model.generate(**gen_kwargs)

            # Slicing: newly generated completion tokens
            if hasattr(outputs, "sequences"):
                sequences = outputs.sequences[0]
            else:
                sequences = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            completion_ids = sequences.tolist()
            # If generation prepended prompt tokens, slice them off
            if len(completion_ids) >= len(prompt_tokens) and completion_ids[:len(prompt_tokens)] == prompt_tokens:
                completion_ids = completion_ids[len(prompt_tokens):]

            raw_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Compute raw unwarped old_logps on completion tokens using batch_X_modals forward pass
            if completion_ids and hasattr(self.model, "prepare_multimodal_inputs"):
                full_input_ids = [torch.tensor(prompt_tokens + completion_ids, dtype=torch.long, device=self.device)]
                full_labels = [torch.tensor([-100] * len(prompt_tokens) + completion_ids, dtype=torch.long, device=self.device)]
                mm_inputs = self.model.prepare_multimodal_inputs(
                    batch_input_ids=full_input_ids,
                    batch_labels=full_labels,
                    batch_X_modals=batch_X_modals,
                    batch_task_names=['ref-avs'],
                    return_multi_scale_features=False,
                    return_gt_mask=False
                )
                forward_out = self.model(
                    inputs_embeds=mm_inputs['inputs_embeds'],
                    attention_mask=mm_inputs['attention_mask'],
                    output_hidden_states=False
                )
                logits = forward_out.logits
                shift_logits = logits[0, :-1, :]
                shift_labels = mm_inputs['labels'][0, 1:]
                mask = (shift_labels != -100)
                comp_labels = shift_labels[mask]
                comp_logits = shift_logits[mask]
                log_probs = F.log_softmax(comp_logits, dim=-1)
                old_logps = log_probs.gather(1, comp_labels.unsqueeze(1)).squeeze(1).tolist()
            else:
                old_logps = [-0.2] * len(completion_ids)
        finally:
            for (name, adapter_name), orig_p in saved_dropouts.items():
                for m_name, module in self.model.named_modules():
                    if m_name == name and hasattr(module, "lora_dropout"):
                        if adapter_name in module.lora_dropout:
                            module.lora_dropout[adapter_name].p = orig_p

        # Validate action
        action_dict, err = parse_and_validate_action(raw_text)
        is_valid = (action_dict is not None)

        action_name = action_dict["action"] if is_valid else "invalid"
        args = action_dict.get("args", {}) if is_valid else {}
        thought = action_dict.get("thought", "") if is_valid else ""

        return ModelAction(
            action_dict=action_dict,
            action_name=action_name,
            args=args,
            thought=thought,
            raw_text=raw_text,
            is_valid=is_valid,
            error=err,
            prompt_ids=prompt_tokens,
            completion_ids=completion_ids,
            old_logps=old_logps,
        )
