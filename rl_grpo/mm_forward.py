from __future__ import annotations
#!/usr/bin/env python3
"""
rl_grpo/mm_forward.py - Shared Multimodal Forward Pass Helper

Unified multimodal embedding and forward pass implementation shared across:
1. SFT Action Format training (scripts/rl_grpo/sft_action_format.py)
2. RL GRPO training policy/reference forward passes (rl_grpo/trainer/grpo_trainer_tgs_v2.py)
3. Multi-turn rollout evaluation (scripts/rl_grpo/infer_pipeline.py)
"""

import os
from typing import Optional, Dict, Any, Tuple, List

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False


def multimodal_forward(
    model: Any,
    input_ids: List[int],
    labels: Optional[List[int]],
    video_tensor: Optional[Any],
    audio_tensor: Optional[Any],
    device: Optional[Any] = None,
    task_name: str = "ref-avs",
) -> Dict[str, Any]:
    """
    Executes the multimodal forward pass on the TGS model with full sequence alignment.
    Handles multimodal feature projection, token insertion (<video>, <audio>),
    and forward computation through the LLaMA backbone.

    Returns:
        {
            "logits": torch.Tensor,
            "loss": Optional[torch.Tensor],
            "shift_logits": torch.Tensor,
            "shift_labels": Optional[torch.Tensor],
            "mask": Optional[torch.Tensor]
        }
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for multimodal_forward.")

    if device is None:
        device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device) if labels is not None else None

    # Normalise multimodal dictionary
    v_tensor = video_tensor.to(device) if (video_tensor is not None and hasattr(video_tensor, "to")) else None
    a_tensor = audio_tensor.to(device) if (audio_tensor is not None and hasattr(audio_tensor, "to")) else None

    batch_X_modals = [{
        "<video>": v_tensor,
        "<audio>": a_tensor,
        "video": v_tensor,
        "audio": a_tensor,
    }]

    if hasattr(model, "prepare_multimodal_inputs"):
        mm_inputs = model.prepare_multimodal_inputs(
            batch_input_ids=[input_ids_t[0]],
            batch_labels=[labels_t[0]] if labels_t is not None else None,
            batch_X_modals=batch_X_modals,
            batch_task_names=[task_name],
            return_multi_scale_features=False,
            return_gt_mask=False
        )

        outputs = model(
            inputs_embeds=mm_inputs["inputs_embeds"],
            attention_mask=mm_inputs["attention_mask"],
            output_hidden_states=False,
            labels=mm_inputs.get("labels", None),
        )

        logits = outputs.logits
        loss = getattr(outputs, "loss", None)

        # Shift tokens for autoregressive token alignment
        shift_logits = logits[0, :-1, :]
        shift_labels = mm_inputs["labels"][0, 1:] if ("labels" in mm_inputs and mm_inputs["labels"] is not None) else None
        mask = (shift_labels != -100) if shift_labels is not None else None

        return {
            "logits": logits,
            "loss": loss,
            "shift_logits": shift_logits,
            "shift_labels": shift_labels,
            "mask": mask,
        }
    else:
        # Text-only baseline / mock forward pass
        outputs = model(input_ids=input_ids_t, labels=labels_t)
        logits = outputs.logits
        loss = getattr(outputs, "loss", None)
        shift_logits = logits[0, :-1, :]
        shift_labels = labels_t[0, 1:] if labels_t is not None else None
        mask = (shift_labels != -100) if shift_labels is not None else None

        return {
            "logits": logits,
            "loss": loss,
            "shift_logits": shift_logits,
            "shift_labels": shift_labels,
            "mask": mask,
        }
