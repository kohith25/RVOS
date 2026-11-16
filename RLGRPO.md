# RL-GRPO Mode for TGS-Agent

This document describes the RL (GRPO) extensions to TGS-Agent. Original TGS files are guarded by the normalized SHA-256 manifest in `tests/test_file_hashes.py`; do not edit those files.

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
| `tools/mock_tools.py` | Explicitly selected mock tools |
| `tools/sam2_tool.py` | SAM2 wrapper (returns masks + logits tuple) |
| `tools/referdino_tool.py` | ReferDINO + GroundingDINO wrappers |
| `kg/build_kg.py` | Per-video multimodal KG builder with batched detection and config-aware cache |
| `kg/audioset_label_map.py` | AudioSet label vocabulary |
| `configs/rl_config.py` | RL hyperparameter config |
| `scripts/check_manifest_unchanged.py` | Prevent unapproved changes to the original-file manifest |
| `tests/test_*.py` | Unit and optional real-weight integration tests |

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
python tools/server.py --port 8088 --grounder gdino
# To use ReferDINO instead: python tools/server.py --port 8088 --grounder referdino
```

### ReferDINO Environment & Setup Notes
ReferDINO is supported through its actual upstream inference code, not a local reimplementation of its model API.

- Repository: [iSEE-Laboratory/ReferDINO](https://github.com/iSEE-Laboratory/ReferDINO).
- The adapter uses upstream `demo_video.py`, `models.build_model(args)`, and `model.infer(samples, [text], [target])`.
- The shipped default config is `configs/ytvos_swinb.yaml`; the shipped default checkpoint is `ckpt/ryt_mevis_swinb.pth`, available from the [ReferDINO Hugging Face model zoo](https://huggingface.co/liangtm/referdino/tree/main).
- The checkpoint is loaded using `checkpoint["model_state_dict"]`, as in upstream `demo_video.py`. Its inference outputs are `pred_logits`, `pred_masks`, and `pred_boxes`. The adapter uses the demo's temporal query scoring and returns the selected query as **one candidate, ID 0**. It does not invent alternatives; SAM2 receives this candidate's per-frame boxes.
- `demo_video.py` accepts `--tracking_alpha` and assigns it to `args.GroundingDINO.tracking_alpha`; the adapter uses the upstream config value (0.1 by default).

Install upstream code and weights in the environment used to run the tool server (`dino`):

```bash
git clone https://github.com/iSEE-Laboratory/ReferDINO.git third_party/ReferDINO
cd third_party/ReferDINO
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
cd models/GroundingDINO/ops
python setup.py build install
cd ../../..
pip install -r requirements.txt
```

Follow the upstream README to place `groundingdino_swint_ogc.pth` and `groundingdino_swinb_cogcoor.pth` under the ReferDINO checkout's `pretrained/` directory. Download `ryt_mevis_swinb.pth` from the model zoo and place it under `third_party/ReferDINO/ckpt/`.

Run real inference on one ten-frame Ref-AVS clip:

  ```bash
  python scripts/smoke_referdino.py \
      --video_dir ./REFAVS/media/<video_id> \
      --expr "the person playing guitar" \
      --repo_dir ./third_party/ReferDINO
  ```

The smoke test exits with an error if the checkout, imports, config, or checkpoint are unavailable. `--mock` is separate and CPU-only; it does not validate real ReferDINO.

The tool server exposes `GET /info` for the configured grounder and `POST /inspect_frames` for observations generated from actual clip audio. Start it with `--grounder referdino` to select this integration.

---

## Step 1 — Cold-Start Action SFT

Build scripted trajectories and fine-tune the model's action format:

```bash
conda activate think

# Build real trajectories from train media; candidate labels use SAM2 mask IoU against train GT only
python scripts/rl_grpo/build_coldstart_data.py \
    --meta_csv_path ./R2AVSBench/RefAVSBench_metadata.csv \
    --num_samples 20 \
    --out_path work_dirs_coldstart/coldstart_data.jsonl

# SFT on action format (saves finetune_weights.bin)
python scripts/rl_grpo/sft_action_format.py \
    --model_name_or_path /path/to/base_llm \
    --avs_ckpt_dir /path/to/tgs_sft_checkpoint \
    --data_path work_dirs_coldstart/coldstart_data.jsonl \
    --output_dir work_dirs_coldstart/sft_action
```

---

## Step 2 — Validate Format Gate (≥ 95% valid)

```bash
python scripts/rl_grpo/check_format_validity.py \
    --ckpt_dir work_dirs_coldstart/sft_action \
    --threshold 0.95
```

---

## Step 3 — GRPO RL Training

```bash
conda activate think
python scripts/rl_grpo/train_rl_tgs.py \
    --model_name_or_path /path/to/base_llm \
    --ckpt_dir work_dirs_coldstart/sft_action \
    --out_dir work_dirs_RL/tgs_grpo \
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

> **Note:** Mock outputs are synthetic and are not research results. Never report mock metrics or mock trajectories as real results; `compare_runs.py` rejects mock result files unless explicitly overridden.

---

## Key Design Constraints

| Constraint | Rule |
|:-----------|:-----|
| Original TGS files | BYTE-IDENTICAL (never modified) — verified by SHA-256 manifest in `tests/test_file_hashes.py` |
| Metric sources | Only `utils/avss_utils.py` — no fabrication (`- 0.03` banned) |
| Mocks | Only via explicit `--mock` flag |
| Missing weights/server | Raises `FileNotFoundError` / `ConnectionError` — no silent fallback |
| Reference model | Default `peft_disable` reference uses no second model; `copy` requires memory for a second 7B model |
| Checkpoint format | `finetune_weights.bin` (LoRA) + `non_lora_trainables.bin` (projectors) |
| SAM2 output | Returns `(masks, logits)` tuple — logits passed to `Eval_Fmeasure` |
| compare_runs mock IDs | Rejected unless `--allow_mock` passed |

---

## Running Tests

```bash
# Full test suite
python -m pytest -q

# Compile-check all new Python files
python -m compileall -q rl_grpo tools kg scripts tests
```

---

## GPU Validation Run Order

Replace the paths and IDs below with the local dataset/checkpoint values. For a five-reference baseline, provide a metadata CSV containing the selected five references; the dispatcher does not expose a `--max_samples` option.

### a. Original pipeline and reproduction check

```bash
python scripts/run_pipeline.py --use_rl false --stage all --test_name test_s \
    --meta_csv_path ./R2AVSBench/refavs_first5.csv \
    --ckpt_dir <TGS_SFT_CHECKPOINT> --avs_ckpt_dir <TGS_SFT_DIR> \
    --out_dir results/original_test_s

python scripts/verify_original_reproduction.py --num_videos 5 --test_name test_s \
    --meta_csv_path ./R2AVSBench/RefAVSBench_metadata.csv \
    --ckpt_dir <TGS_SFT_CHECKPOINT> --avs_ckpt_dir <TGS_SFT_DIR> \
    --out_dir_pipeline results/verify_pipeline
```

### b. Real ReferDINO inference

```bash
python scripts/smoke_referdino.py --video_dir ./REFAVS/media/<video_id> \
    --expr "the object making a sound on the left of the guitar" \
    --repo_dir ./third_party/ReferDINO
```

### c. Build KG for two real videos

```bash
python kg/build_kg.py --video_id <train_uid_1> --media_dir ./REFAVS/media \
    --meta_csv_path ./R2AVSBench/RefAVSBench_metadata.csv --kg_dir kg_cache
python kg/build_kg.py --video_id <train_uid_2> --media_dir ./REFAVS/media \
    --meta_csv_path ./R2AVSBench/RefAVSBench_metadata.csv --kg_dir kg_cache
```

### d. Cold-start SFT and format gate

Keep the tool server running in the `dino` environment. Candidate choice uses SAM2 masks and train GT only; the GT IoU is stored as training metadata, not in agent prompts.

```bash
python scripts/rl_grpo/build_coldstart_data.py \
    --meta_csv_path ./R2AVSBench/RefAVSBench_metadata.csv \
    --num_samples 20 --out_path work_dirs_coldstart/coldstart_data.jsonl
python scripts/rl_grpo/sft_action_format.py \
    --model_name_or_path <LLAMA_2_7B_CHAT_DIR> \
    --avs_ckpt_dir <TGS_SFT_DIR> \
    --data_path work_dirs_coldstart/coldstart_data.jsonl \
    --output_dir work_dirs_coldstart/sft_action
python scripts/rl_grpo/check_format_validity.py \
    --ckpt_dir work_dirs_coldstart/sft_action \
    --avs_ckpt_dir <TGS_SFT_DIR> \
    --model_name_or_path <LLAMA_2_7B_CHAT_DIR> \
    --threshold 0.95
```

Proceed to GRPO only if the real format-validity run reaches at least 0.95.

### e. Five GRPO steps, inference, and comparison

```bash
python scripts/rl_grpo/train_rl_tgs.py \
    --model_name_or_path <LLAMA_2_7B_CHAT_DIR> \
    --ckpt_dir work_dirs_coldstart/sft_action \
    --out_dir work_dirs_RL/tgs_grpo_5steps \
    --max_steps 5 --num_generations 4 \
    --grounder referdino --use_kg true

python scripts/rl_grpo/infer_pipeline.py \
    --ckpt_dir work_dirs_RL/tgs_grpo_5steps \
    --avs_ckpt_dir <TGS_SFT_DIR> \
    --out_dir results/rl_test_s \
    --test_name test_s --grounder referdino --use_kg true

python scripts/compare_runs.py --run1 results/original_test_s \
    --run2 results/rl_test_s --label1 "Original TGS" --label2 "RL GRPO"
```

These are real-weight instructions, not a claim that GPU evaluation has already run. The current development tests use CPU mocks and fixtures.

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
- **Step-0 policy ratio**: logs the measured maximum `|exp(current_logp - rollout_logp) - 1|` before the first optimizer update; LoRA dropout is disabled while recomputing those log probabilities.

---

## Acknowledgement

RL-GRPO mode authored as an extension to [TGS-Agent](https://github.com/jasongief/TGS-Agent). See original README for base model setup.
