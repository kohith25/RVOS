# RL-GRPO Mode for TGS-Agent

This document describes the **RL (GRPO) mode** added on top of the original TGS-Agent pipeline. All original TGS files remain byte-identical; only the files listed below are added/modified.

> **Important:** All metrics are computed by the original `utils/avss_utils.py` functions (`mask_iou`, `Eval_Fmeasure`, `metric_s_for_null_batch`). No metric fabrication, no silent fallbacks.

---

## Files Added / Modified

| Path | Description |
|:-----|:------------|
| `scripts/run_pipeline.py` | Unified dispatcher (`--use_rl false\|true`) |
| `scripts/compare_runs.py` | Side-by-side run comparison |
| `scripts/rl_grpo/infer_pipeline.py` | Multi-turn RL rollout + evaluation |
| `scripts/rl_grpo/train_rl_tgs.py` | GRPO training entry point |
| `scripts/rl_grpo/build_coldstart_data.py` | Build scripted cold-start trajectories |
| `scripts/rl_grpo/sft_action_format.py` | Action-format SFT before GRPO |
| `scripts/rl_grpo/check_format_validity.py` | Format validity gate (≥95%) |
| `rl_grpo/arguments.py` | `RLGRPOArguments` dataclass |
| `rl_grpo/build_model.py` | Model builder (mirrors original `.sh`) |
| `rl_grpo/model_adapter.py` | `TGSModelAdapter` (sample / build prompts) |
| `rl_grpo/rewards.py` | Episodic reward functions |
| `rl_grpo/trajectory.py` | Trajectory / rollout manager |
| `rl_grpo/constants.py` | Action tokens, observation formatter |
| `rl_grpo/trainer/grpo_trainer_tgs_v2.py` | GRPO trainer with PPO clipping + k3 KL |
| `tools/server.py` | Tool server (ReferDINO + SAM2, `--grounder` flag) |
| `tools/client.py` | Tool client with disk cache + mock fallback |
| `tools/sam2_tool.py` | SAM2 wrapper (returns masks + logits tuple) |
| `tools/referdino_tool.py` | ReferDINO + GroundingDINO wrappers |
| `tools/mock_tools.py` | Mock tools for CPU-only dev/CI |
| `kg/build_kg.py` | Knowledge graph builder (AudioSet labels) |
| `kg/audioset_label_map.py` | AudioSet label vocabulary |
| `configs/rl_config.py` | RL hyperparameter config |
| `tests/test_*.py` | Full test suite (58 passed, 5 skipped) |

---

## Architecture

```
scripts/run_pipeline.py          # Unified dispatcher (--use_rl false|true)
├── --use_rl false               # Original TGS pipeline (byte-identical)
└── --use_rl true
    ├── scripts/rl_grpo/infer_pipeline.py    # Multi-turn rollout + evaluation
    ├── rl_grpo/trainer/grpo_trainer_tgs_v2.py  # GRPO trainer
    ├── rl_grpo/rewards.py                   # Reward functions
    ├── rl_grpo/trajectory.py                # Trajectory / rollout manager
    ├── tools/server.py                      # Tool server (ReferDINO + SAM2)
    └── kg/build_kg.py                       # Knowledge graph (AudioSet labels)
```

---

## Step 0 — Start the Tool Server (in `dino` conda env)

```bash
conda activate dino
python tools/server.py --port 8088 --grounder referdino
# To use GroundingDINO instead: --grounder grounding_dino
```

---

## Step 1 — Cold-Start Action SFT

Build scripted trajectories and fine-tune the model's action format:

```bash
conda activate think

# Build 4-turn scripted trajectories from the train split
python scripts/rl_grpo/build_coldstart_data.py \
    --meta_csv_path data/ref_avs_meta.csv \
    --out_jsonl work_dirs_coldstart/coldstart_data.jsonl

# SFT on action format (saves finetune_weights.bin)
python scripts/rl_grpo/sft_action_format.py \
    --model_name_or_path /path/to/base_llm \
    --ckpt_dir /path/to/tgs_sft_checkpoint \
    --data_path work_dirs_coldstart/coldstart_data.jsonl \
    --output_dir work_dirs_coldstart/sft_action
```

---

## Step 2 — Validate Format Gate (≥ 95% valid)

```bash
python scripts/rl_grpo/check_format_validity.py \
    --model_path work_dirs_coldstart/sft_action \
    --data_path work_dirs_coldstart/coldstart_data.jsonl \
    --threshold 0.95
```

---

## Step 3 — GRPO RL Training

```bash
conda activate think
python scripts/rl_grpo/train_rl_tgs.py \
    --model_name_or_path /path/to/base_llm \
    --ckpt_dir work_dirs_coldstart/sft_action \
    --avs_ckpt_dir /path/to/tgs_sft_checkpoint \
    --output_dir work_dirs_RL/tgs_grpo \
    --meta_csv_path data/ref_avs_meta.csv \
    --tool_server_url http://127.0.0.1:8088 \
    --num_generations 4 \
    --max_steps 1000 \
    --learning_rate 5e-6 \
    --beta 0.04
```

---

## Step 4 — Inference and Evaluation

### `--use_rl false` (original pipeline)

```bash
conda activate think
python scripts/run_pipeline.py \
    --use_rl false \
    --stage all \
    --test_name test_s \
    --ckpt_dir /path/to/tgs_sft_checkpoint \
    --out_dir results/original_test_s
```

### `--use_rl true` (RL policy)

```bash
conda activate think
python scripts/run_pipeline.py \
    --use_rl true \
    --stage all \
    --test_name test_s \
    --ckpt_dir work_dirs_RL/tgs_grpo \
    --out_dir results/rl_run_test_s \
    --tool_server_url http://127.0.0.1:8088
```

**Dry-run preview** (no files written, no model loaded):

```bash
python scripts/run_pipeline.py --use_rl false --dry_run --stage all --test_name test_s
python scripts/run_pipeline.py --use_rl true  --dry_run --stage all --test_name test_s
```

---

## Step 5 — Compare Runs

```bash
python scripts/compare_runs.py \
    --run1 results/original_test_s \
    --run2 results/rl_run_test_s \
    --label1 "Original TGS" \
    --label2 "RL GRPO"
# Add --allow_mock to include mock/synthetic video IDs (dev/debug only)
```

---

## Mock Mode (CPU-only, no weights)

For development and CI without GPU or model weights, add `--mock`:

```bash
python scripts/run_pipeline.py \
    --use_rl true --mock --stage infer \
    --test_name test_s --out_dir /tmp/rl_mock_out
```

> **Note:** Mock metrics are placeholder zeros and cannot be compared against real results without `--allow_mock`. Never report mock metrics as real results.

---

## Key Design Constraints

| Constraint | Rule |
|:-----------|:-----|
| Original TGS files | BYTE-IDENTICAL (never modified) — verified by SHA-256 manifest in `tests/test_file_hashes.py` |
| Metric sources | Only `utils/avss_utils.py` — no fabrication (`- 0.03` banned) |
| Mocks | Only via explicit `--mock` flag |
| Missing weights/server | Raises `FileNotFoundError` / `ConnectionError` — no silent fallback |
| Reference model | Frozen copy of SFT weights; KL ≈ 0 at step 0 |
| Checkpoint format | `finetune_weights.bin` (LoRA) + `non_lora_trainables.bin` (projectors) |
| SAM2 output | Returns `(masks, logits)` tuple — logits passed to `Eval_Fmeasure` |
| compare_runs mock IDs | Rejected unless `--allow_mock` passed |

---

## Running Tests

```bash
# Full test suite (58 passed, 5 skipped — skips require PyTorch/GPU)
python -m pytest TGS-Agent/tests/ -v

# Compile-check all new Python files
python -m compileall TGS-Agent/rl_grpo TGS-Agent/tools TGS-Agent/kg \
    TGS-Agent/scripts/rl_grpo TGS-Agent/tests
```

---

## GRPO Trainer Details

The trainer (`rl_grpo/trainer/grpo_trainer_tgs_v2.py`) implements:

- **Per-group advantage normalization**: `A = (R - mean(R)) / (std(R) + 1e-4)` within each group of G rollouts
- **PPO clipped surrogate loss**: `clip(ratio, 1-ε, 1+ε) × A`
- **k3 KL estimator**: `exp(log_ref - log_curr) - (log_ref - log_curr) - 1`
- **Exact token alignment**: loss only on completion tokens; prompt + observation tokens masked out (`-100`)
- **Multimodal forward**: uses `model.prepare_multimodal_inputs()` with `batch_X_modals` for video/audio
- **MM module freezing**: projectors, QFormers frozen by default; enabled with `--train_mm_modules`
- **Gradient clipping**: `max_grad_norm=1.0` via `clip_grad_norm_`
- **Checkpoint format**: saves `finetune_weights.bin` (LoRA) + `non_lora_trainables.bin` (projectors/QFormers) exactly as expected by `inference_hyper_lora.py`

---

## Acknowledgement

RL-GRPO mode authored as an extension to [TGS-Agent](https://github.com/jasongief/TGS-Agent). See original README for base model setup.
