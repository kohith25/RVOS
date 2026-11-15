from __future__ import annotations
#!/usr/bin/env python3
"""
scripts/rl_grpo/sft_action_format.py - LoRA SFT for Ref-Thinker Action Format Conditioning

Performs supervised fine-tuning of Ref-Thinker LoRA adapters on cold-start action trajectories:
- Loads REAL video/audio tensors per sample through the repo's preprocessing path (NEVER zeros in real mode).
- In --mock mode, dummy tensors are permitted and a MOCK_DO_NOT_USE marker file is created.
- Computes cross-entropy loss ONLY on completion tokens (prompt tokens masked with -100).
- Uses shared multimodal_forward from rl_grpo.mm_forward.
- Saves weights in original split format:
    finetune_weights.bin     (LoRA tensors only)
    non_lora_trainables.bin  (projector/Q-Former/query tensors only)
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Optional
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = object
    AdamW = None
    HAS_TORCH = False

from rl_grpo.build_model import build_tgs_model
from rl_grpo.mm_forward import multimodal_forward


def load_real_media_tensors(video_path: str, audio_path: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Loads and preprocesses real video frames and audio using standard repo preprocessing.
    Raises FileNotFoundError if media is missing.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to load media tensors.")

    # 1. Video Frames Loading
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path not found: {video_path}")

    if os.path.isdir(video_path):
        frame_files = sorted([
            os.path.join(video_path, f) for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])[:10]
    else:
        frame_files = [video_path]

    if not frame_files:
        raise FileNotFoundError(f"No video frames found in {video_path}")

    # Load 10 frames into tensor [10, 3, 224, 224] normalized
    frames_pil = [Image.open(f).convert("RGB").resize((224, 224)) for f in frame_files]
    while len(frames_pil) < 10:
        frames_pil.append(frames_pil[-1])

    frames_np = np.stack([np.array(im, dtype=np.float32).transpose(2, 0, 1) / 255.0 for im in frames_pil])
    video_tensor = torch.tensor(frames_np, dtype=torch.float32)

    # 2. Audio Waveform Loading
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        from dataset.audio_processor import preprocess
        audio_tensor = preprocess(audio_path)
    except Exception:
        import wave
        with wave.open(audio_path, "rb") as wf:
            n_frames = wf.getnframes()
            sr = wf.getframerate()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() > 1:
                samples = samples.reshape(-1, wf.getnchannels()).mean(axis=1)
        # Pad/truncate to 10 seconds of 16kHz audio
        target_len = sr * 10
        if len(samples) < target_len:
            samples = np.pad(samples, (0, target_len - len(samples)))
        else:
            samples = samples[:target_len]
        audio_tensor = torch.tensor(samples, dtype=torch.float32).unsqueeze(0)

    return video_tensor, audio_tensor


def get_parser():
    parser = argparse.ArgumentParser(description="Action Format LoRA SFT for Ref-Thinker")
    parser.add_argument("--data_path", type=str, default="data/coldstart_trajectories.jsonl")
    parser.add_argument("--output_dir", type=str, default="./work_dirs_RL/sft_format")
    parser.add_argument("--model_name_or_path", type=str, default="./pretrained_weights/Llama-2-7b-chat-hf")
    parser.add_argument("--avs_ckpt_dir", type=str, default="./results_real/epochs6_lr1e-4_bs4_gradacc8_lora_r8alpha16dropout0.05")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mock", action="store_true", help="Run with mock model and dummy tensors on CPU")
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args)


def save_sft_checkpoint(model: Any, output_dir: str):
    """
    Saves trained weights in the standard split format:
    - finetune_weights.bin: LoRA keys
    - non_lora_trainables.bin: projector/Q-Former/query keys
    """
    os.makedirs(output_dir, exist_ok=True)
    lora_state = {}
    non_lora_state = {}

    critical_mm = ["projector", "qformer", "visual_query", "audio_query", "query_generator"]

    if hasattr(model, "state_dict"):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        state_dict = {}

    for k, v in state_dict.items():
        tensor_val = v.cpu() if hasattr(v, "cpu") else v
        if "lora" in k.lower():
            lora_state[k] = tensor_val
        elif any(sub in k.lower() for sub in critical_mm):
            non_lora_state[k] = tensor_val

    weights_file = os.path.join(output_dir, "finetune_weights.bin")
    non_lora_file = os.path.join(output_dir, "non_lora_trainables.bin")

    torch.save(lora_state, weights_file)
    torch.save(non_lora_state, non_lora_file)
    print(f"[SFT Checkpoint] Saved {len(lora_state)} LoRA tensors to {weights_file}")
    print(f"[SFT Checkpoint] Saved {len(non_lora_state)} projector/trainable tensors to {non_lora_file}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Ref-Thinker Action Format LoRA SFT")
    print(f"  Data: {args.data_path} | Output: {args.output_dir} | Mock: {args.mock}")
    print("=" * 65)

    # 1. Load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Cold-start data file not found at: {args.data_path}")

    turns = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))

    print(f"[SFT] Loaded {len(turns)} training turns.")

    # 2. Build model
    if args.mock:
        model, _, tokenizer = build_tgs_model(use_mock=True)
        print("[SFT] Running in MOCK mode. Writing MOCK_DO_NOT_USE marker file...")
        marker_path = os.path.join(args.output_dir, "MOCK_DO_NOT_USE")
        with open(marker_path, "w") as fp:
            fp.write("This checkpoint was generated in --mock mode for testing purposes only.\n")

        # Mock tensor generation is permitted ONLY in mock mode
        mock_lora = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(4, 4)}
        mock_proj = {"model.model.vision_projector.weight": torch.randn(4, 4)}
        torch.save(mock_lora, os.path.join(args.output_dir, "finetune_weights.bin"))
        torch.save(mock_proj, os.path.join(args.output_dir, "non_lora_trainables.bin"))
        print(f"[SFT SUCCESS] Mock split checkpoints saved to {args.output_dir}")
        return 0

    # Non-mock path (requires real PyTorch and GPU)
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for real SFT training.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, tokenizer = build_tgs_model(
        model_name_or_path=args.model_name_or_path,
        avs_ckpt_dir=args.avs_ckpt_dir,
        device=str(device),
        use_mock=False,
    )
    model.train()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer.zero_grad()

    random.seed(42)

    for epoch in range(args.epochs):
        random.shuffle(turns)
        total_loss = 0.0

        for idx, turn in enumerate(turns):
            prompt = turn["prompt_text"]
            completion = turn["completion_text"]
            video_path = turn.get("video_path", "")
            audio_path = turn.get("audio_path", "")

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            comp_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]

            full_ids = prompt_ids + comp_ids
            # Mask prompt tokens with -100 so loss is computed ONLY on completion JSON tokens
            labels = [-100] * len(prompt_ids) + comp_ids

            # Load REAL media tensors — NEVER zeros in real mode!
            video_tensor, audio_tensor = load_real_media_tensors(video_path, audio_path)

            fwd_out = multimodal_forward(
                model=model,
                input_ids=full_ids,
                labels=labels,
                video_tensor=video_tensor,
                audio_tensor=audio_tensor,
                device=device,
            )

            loss = fwd_out["loss"]
            if loss is not None and loss.requires_grad:
                scaled_loss = loss / args.grad_accum_steps
                scaled_loss.backward()
                total_loss += loss.item()

            if (idx + 1) % args.grad_accum_steps == 0 or (idx + 1) == len(turns):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = total_loss / max(1, len(turns))
        print(f"[SFT] Epoch {epoch + 1}/{args.epochs} | Avg Loss: {avg_loss:.4f}")

    # Save trained split weights
    save_sft_checkpoint(model, args.output_dir)
    print(f"[SFT SUCCESS] Format SFT training complete. Checkpoints saved to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
