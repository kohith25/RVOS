"""
configs/rl_config.py
====================
Extended training arguments for TGS-Agent + VideoSEG-O3 GRPO integration.

KEY FLAG
--------
    use_rl : bool  (default False)
        False → standard TGS-Agent training (original code path, unchanged).
        True  → GRPO / RL training via rl_grpo/.

LOSS FORMULA (when use_rl=True)
---------------------------------
    L_total = L_GRPO + lambda_seg * L_seg + lambda_reason * L_reason

    - L_GRPO    : policy gradient loss (VideoSEG-O3-style GRPO with KL penalty)
    - L_seg     : Dice + focal-BCE on SAM2 masks  (lambda_seg=0 freezes SAM2)
    - L_reason  : supervised CE on <think>/<action>/<answer> tokens
    - L_RAG     : INTENTIONALLY OMITTED. Evidence reward R_e teaches the policy
                  when to retrieve; the KG/retriever is treated as a fixed oracle.

COMPOSITE REWARD (R)
---------------------
    R = wf*Rf + wt*Rt + wm*Rm + we*Re + wp*Rp - wc*C

    Rf : Format reward       — valid structured output
    Rt : Temporal reward     — informative frame inspection
    Rm : Segmentation reward — IoU/Dice/J&F mask quality
    Re : Evidence reward     — correct retrieval and usage
    Rp : Progressive reward  — monotonic localization improvement
    C  : Tool cost penalty   — avoids unnecessary tool calls
"""

from dataclasses import dataclass, field
from typing import Optional

import transformers

# Re-use the original TGS-Agent TrainingArguments so the original training
# pipeline never changes.
from configs.unified_config import TrainingArguments as TGSTrainingArguments


@dataclass
class RLTrainingArguments(TGSTrainingArguments):
    """
    Drop-in replacement for TGS-Agent's TrainingArguments that adds an RL
    section.  When use_rl=False every RL field is ignored and the original
    training code runs exactly as before.
    """

    # ------------------------------------------------------------------ #
    # MASTER SWITCH                                                        #
    # ------------------------------------------------------------------ #
    use_rl: bool = field(
        default=False,
        metadata={
            "help": (
                "Master RL switch.  False → original TGS-Agent training "
                "(unchanged).  True → GRPO training via rl_grpo/."
            )
        },
    )

    # ------------------------------------------------------------------ #
    # GRPO / POLICY SETTINGS                                              #
    # ------------------------------------------------------------------ #
    rl_max_turn: int = field(
        default=3,
        metadata={"help": "Maximum reasoning turns per episode."},
    )
    rl_num_generations: int = field(
        default=4,
        metadata={"help": "Number of trajectories sampled per input (G in GRPO)."},
    )
    rl_max_steps: int = field(
        default=1000,
        metadata={"help": "Total RL training steps."},
    )
    rl_learning_rate: float = field(
        default=3e-6,
        metadata={"help": "Learning rate for the RL optimiser."},
    )
    rl_weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for the RL optimiser."},
    )
    rl_grad_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient norm clipping for RL updates."},
    )
    rl_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for the RL scheduler."},
    )
    rl_beta: float = field(
        default=0.04,
        metadata={"help": "KL penalty coefficient (β in GRPO)."},
    )
    rl_kl_approximator: str = field(
        default="k3",
        metadata={
            "help": (
                "KL approximator type: k1, k3 (default, VideoSEG-O3), "
                "kimikl, fullkimi."
            )
        },
    )
    rl_use_kl: bool = field(
        default=True,
        metadata={"help": "Whether to include KL penalty in the RL loss."},
    )

    # ------------------------------------------------------------------ #
    # GENERATION / SAMPLING                                               #
    # ------------------------------------------------------------------ #
    rl_temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for trajectory generation."},
    )
    rl_top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling top-p for trajectory generation."},
    )
    rl_max_action_tokens: int = field(
        default=512,
        metadata={"help": "Max new tokens per turn during rollout."},
    )
    rl_max_prompt_length: int = field(
        default=8192,
        metadata={"help": "Max token length of the prompt fed to the model."},
    )
    rl_max_completion_length: int = field(
        default=512,
        metadata={"help": "Max new tokens for the full completion."},
    )

    # ------------------------------------------------------------------ #
    # LOSS WEIGHTS  (L_total = L_GRPO + λ_seg·L_seg + λ_reason·L_reason) #
    # ------------------------------------------------------------------ #
    rl_lambda_seg: float = field(
        default=0.2,
        metadata={
            "help": (
                "Weight for segmentation loss L_seg (Dice + focal-BCE). "
                "Set to 0 to freeze SAM2 components."
            )
        },
    )
    rl_lambda_reason: float = field(
        default=0.2,
        metadata={
            "help": (
                "Weight for supervised reasoning / action-generation loss "
                "L_reason (CE on <think>/<action>/<answer> tokens)."
            )
        },
    )
    rl_rec_loss_ratio: float = field(
        default=1.0,
        metadata={"help": "Scaling ratio applied to L_GRPO (RL policy loss)."},
    )
    rl_res_loss_ratio: float = field(
        default=0.2,
        metadata={"help": "Scaling ratio applied to auxiliary mask losses."},
    )

    # ------------------------------------------------------------------ #
    # REWARD WEIGHTS  R = wf·Rf + wt·Rt + wm·Rm + we·Re + wp·Rp - wc·C  #
    # ------------------------------------------------------------------ #
    rl_reward_wf: float = field(default=1.0, metadata={"help": "Format reward weight."})
    rl_reward_wt: float = field(default=1.0, metadata={"help": "Temporal reward weight."})
    rl_reward_wm: float = field(default=1.0, metadata={"help": "Segmentation reward weight."})
    rl_reward_we: float = field(default=1.0, metadata={"help": "Evidence reward weight."})
    rl_reward_wp: float = field(default=1.0, metadata={"help": "Progressive reward weight."})
    rl_reward_wc: float = field(default=0.25, metadata={"help": "Tool cost penalty weight."})
    rl_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Global reward scaling factor (applied to all reward components)."},
    )

    # ------------------------------------------------------------------ #
    # REWARD FUNCTION SELECTION                                           #
    # ------------------------------------------------------------------ #
    rl_reward_funcs: str = field(
        default="format,temporal,segmentation,evidence,progressive",
        metadata={
            "help": (
                "Comma-separated list of reward functions to enable. "
                "Choices: format, temporal, segmentation, evidence, progressive, bbox."
            )
        },
    )

    # ------------------------------------------------------------------ #
    # COMPONENT FREEZE FLAGS                                              #
    # ------------------------------------------------------------------ #
    rl_freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze the LLM backbone during RL training."},
    )
    rl_freeze_visual: bool = field(
        default=True,
        metadata={"help": "Freeze the visual encoder during RL training."},
    )
    rl_freeze_audio: bool = field(
        default=True,
        metadata={"help": "Freeze the audio encoder during RL training."},
    )
    rl_freeze_rag: bool = field(
        default=True,
        metadata={"help": "Always True: the KG/RAG retriever is never trained."},
    )

    # ------------------------------------------------------------------ #
    # LOGGING / CHECKPOINT                                                #
    # ------------------------------------------------------------------ #
    rl_logging_steps: int = field(default=10, metadata={"help": "Log RL metrics every N steps."})
    rl_save_steps: int = field(default=200, metadata={"help": "Save RL checkpoint every N steps."})
    rl_use_mask_logps: bool = field(
        default=False,
        metadata={"help": "Inject per-mask log-probs into the [SEG] token positions (expensive)."},
    )
    rl_if_use_visualization: bool = field(
        default=False,
        metadata={"help": "Visualize rollout samples during RL training."},
    )
    rl_vis_interval: int = field(
        default=20,
        metadata={"help": "Visualize every N steps (requires rl_if_use_visualization=True)."},
    )
    rl_skip_special_tokens: bool = field(
        default=False,
        metadata={"help": "Skip special tokens when decoding completions for reward computation."},
    )
