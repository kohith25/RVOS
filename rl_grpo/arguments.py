from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RLGRPOArguments:
    enabled: bool = field(default=False)
    max_turn: int = field(default=3)
    num_generations: int = field(default=4)
    max_steps: int = field(default=1000)
    learning_rate: float = field(default=3e-6)
    weight_decay: float = field(default=0.0)
    max_grad_norm: float = field(default=1.0)
    warmup_steps: int = field(default=0)
    beta: float = field(default=0.04)

    # Paths and reproducibility
    output_dir: str = field(default="./work_dirs_RL/tgs_grpo")
    seed: int = field(default=42)
    ckpt_dir: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)

    # Reward weights: R = wf*Rf + wt*Rt + wm*Rm + we*Re + wp*Rp - wc*C
    reward_wf: float = field(default=1.0)
    reward_wt: float = field(default=1.0)
    reward_wm: float = field(default=1.0)
    reward_we: float = field(default=1.0)
    reward_wp: float = field(default=1.0)
    reward_wc: float = field(default=0.25)

    # Optional supervised losses in the mixed objective.
    lambda_seg: float = field(default=0.0)
    lambda_reason: float = field(default=0.0)

    # Sampling / prompt behavior.
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.95)
    max_action_tokens: int = field(default=96)
    clip_epsilon: float = field(default=0.2)

    # Logging and checkpointing.
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)

    # Multimodal freezing
    train_mm_modules: bool = field(default=False)

    # Reference model mode:
    #   "peft_disable" = use adapter-disable context manager (no extra GPU memory).
    #   "copy" = frozen separate copy of the full model (doubles memory).
    ref_mode: str = field(default="peft_disable")

    # Whether the KG retrieve_evidence action is enabled
    use_kg: bool = field(default=True)
