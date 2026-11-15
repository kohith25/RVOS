from __future__ import annotations
#!/usr/bin/env python3
"""
kg/build_kg.py - Multimodal Knowledge Graph Builder (Real Media-Driven)

Constructs an episodic multimodal knowledge graph per video derived strictly from actual media:
- Visual entities / tracklets: detected objects linked across 10 frames via spatial IoU tracking.
- Per-second motion energy: computed from real pixel differences within tracklet bounding boxes.
- Audio events: top-3 AudioSet labels per second from BEATs acoustic model.
- Per-second RMS loudness: computed from actual audio waveform.
- Sounding edges: Pearson correlation between entity motion energy and audio loudness.
- Text referring entity: matched to tracklet labels by semantic / token overlap.
- Schema versioning & config hashing for cache validity.
- Strict error handling: raises FileNotFoundError on missing media or weights (no silent fallback).
"""

import os
import sys
import json
import re
import csv
import wave
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kg.audioset_label_map import get_audioset_label, AUDIOSET_LABELS

SCHEMA_VERSION = 2
DEFAULT_KG_DIR = "kg_cache"


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """Computes a SHA-256 hash of the KG builder configuration for cache validation."""
    serialized = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def resolve_media_paths(video_id: str, media_dir: str = "./REFAVS/media") -> Tuple[List[str], str]:
    """
    Resolves the 10 video frame paths and audio path for a given video ID.
    Follows Ref-AVSBench directory layout:
        frames: <media_dir>/<vid>/frames/<0..9>.jpg (or .png)
        audio:  <media_dir>/<vid>/audio.wav (or media_cross)
    Raises FileNotFoundError if media is missing.
    """
    vid = video_id.rsplit("_", 2)[0] if "_" in video_id else video_id

    # Check potential frame directories
    candidates = [
        os.path.join(media_dir, vid, "frames"),
        os.path.join(media_dir, video_id, "frames"),
        os.path.join(media_dir, vid),
    ]

    frame_dir = None
    for cand in candidates:
        if os.path.isdir(cand):
            frame_dir = cand
            break

    if frame_dir is None:
        raise FileNotFoundError(
            f"Video frames directory not found for video_id '{video_id}' in '{media_dir}'. "
            f"Checked paths: {candidates}"
        )

    # Collect 10 frame image paths
    frame_paths = []
    for i in range(10):
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(frame_dir, f"{i}{ext}")
            if os.path.exists(p):
                frame_paths.append(p)
                break
            # Try frame_000i formatting
            p_pad = os.path.join(frame_dir, f"frame_{i:04d}{ext}")
            if os.path.exists(p_pad):
                frame_paths.append(p_pad)
                break

    if len(frame_paths) < 10:
        # Fallback to sorted image files in directory
        all_imgs = sorted([
            os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        if len(all_imgs) >= 10:
            frame_paths = all_imgs[:10]

    if len(frame_paths) < 10:
        raise FileNotFoundError(
            f"Expected at least 10 frames for video '{video_id}' in '{frame_dir}', found {len(frame_paths)}."
        )

    # Check potential audio paths
    audio_candidates = [
        os.path.join(media_dir, vid, "audio.wav"),
        os.path.join(media_dir, video_id, "audio.wav"),
        os.path.join(os.path.dirname(media_dir), "media_cross", vid, "audio.wav"),
    ]

    audio_path = None
    for cand in audio_candidates:
        if os.path.exists(cand):
            audio_path = cand
            break

    if audio_path is None:
        raise FileNotFoundError(
            f"Audio file 'audio.wav' not found for video_id '{video_id}'. "
            f"Checked paths: {audio_candidates}"
        )

    return frame_paths, audio_path


def read_audio_waveform(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    Reads an audio file into a 1D float32 numpy array normalized to [-1.0, 1.0].
    Handles mono, stereo, and variable bit depths robustly.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        pass

    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 4:
                samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            return samples, sr
    except Exception as e:
        raise RuntimeError(f"Failed to read audio waveform from '{audio_path}': {e}")


def compute_audio_loudness_per_second(samples: np.ndarray, sr: int, num_seconds: int = 10) -> List[float]:
    """
    Computes RMS loudness per second over the waveform, normalized to [0.0, 1.0].
    """
    loudness = []
    samples_per_sec = max(1, sr)
    for sec in range(num_seconds):
        start = sec * samples_per_sec
        end = min(len(samples), (sec + 1) * samples_per_sec)
        if start < len(samples) and end > start:
            chunk = samples[start:end]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
        else:
            rms = 0.0
        # Scale RMS into [0.0, 1.0] range
        val = min(1.0, rms * 5.0)
        loudness.append(round(val, 4))
    return loudness


def compute_tracklet_motion_energy(
    frame_images: List[Image.Image],
    boxes: List[List[float]],
) -> List[float]:
    """
    Computes per-second motion energy from real pixel differences within tracklet bounding boxes.
    Normalized to [0.0, 1.0].
    """
    num_frames = len(frame_images)
    if num_frames < 2:
        return [0.0] * num_frames

    motion_diffs = []
    for t in range(num_frames - 1):
        b1 = boxes[t] if t < len(boxes) else [0, 0, 1, 1]
        b2 = boxes[t + 1] if (t + 1) < len(boxes) else b1

        # Use the union bounding box of consecutive frames
        x1 = min(b1[0], b2[0])
        y1 = min(b1[1], b2[1])
        x2 = max(b1[2], b2[2])
        y2 = max(b1[3], b2[3])

        # Avoid zero-area boxes
        if x2 <= x1 or y2 <= y1:
            motion_diffs.append(0.0)
            continue

        im1 = frame_images[t].convert("L")
        im2 = frame_images[t + 1].convert("L")
        w, h = im1.size

        px1 = int(np.clip(x1 * w, 0, w - 1))
        py1 = int(np.clip(y1 * h, 0, h - 1))
        px2 = int(np.clip(x2 * w, px1 + 1, w))
        py2 = int(np.clip(y2 * h, py1 + 1, h))

        crop1 = np.array(im1.crop((px1, py1, px2, py2)), dtype=np.float32)
        crop2 = np.array(im2.crop((px1, py1, px2, py2)), dtype=np.float32)

        if crop1.size == 0 or crop2.size == 0 or crop1.shape != crop2.shape:
            motion_diffs.append(0.0)
            continue

        # Normalized pixel difference in [0.0, 1.0]
        diff = float(np.mean(np.abs(crop2 - crop1))) / 255.0
        motion_diffs.append(round(diff, 4))

    # Pad the final frame transition to make 10 entries (1 per second)
    if motion_diffs:
        motion_diffs.append(motion_diffs[-1])
    else:
        motion_diffs = [0.0] * num_frames

    return motion_diffs


def extract_noun_phrases(text: str) -> List[str]:
    """Simple regex/heuristic noun phrase extractor from referring expression."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop_words = {"the", "and", "that", "this", "with", "from", "for", "are", "was", "were", "which", "playing", "performing"}
    nouns = [w for w in words if w not in stop_words]
    return nouns


def get_object_vocabulary(meta_csv_path: Optional[str] = None) -> List[str]:
    """Extracts or returns the Ref-AVSBench object vocabulary."""
    if meta_csv_path and os.path.exists(meta_csv_path):
        try:
            vocab = set()
            with open(meta_csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        uid = row[1]
                        parts = uid.rsplit("_", 2)
                        if len(parts) >= 3 and parts[1]:
                            vocab.add(parts[1].lower().replace("-", " "))
            if vocab:
                return sorted(list(vocab))
        except Exception:
            pass

    # Standard AVSBench 51 classes
    return [
        "accordion", "airplane", "ambulance", "baby", "banjo", "bassoon",
        "bell", "bongo", "car", "cat", "cello", "clarinet", "dog",
        "donkey", "drum", "duck", "electric guitar", "elephant", "flute",
        "french horn", "frog", "giant panda", "gong", "guitar", "gun",
        "harmonica", "harp", "helicopter", "horse", "keyboard", "lawn mower",
        "lion", "mandolin", "marimba", "motorcycle", "oboe", "person",
        "piano", "police car", "saxophone", "sheep", "singing", "table tennis",
        "tank", "tiger", "tractor", "train", "trombone", "truck", "trumpet",
        "tuba", "violin", "whale", "xylophone"
    ]


def run_grounding_dino_on_frames(
    frame_paths: List[str],
    text_prompt: str,
    detector: Optional[Any] = None,
    box_threshold: float = 0.15,
    text_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Runs visual object detection on 10 frames and links detections into tracklets via IoU.
    If detector is None, attempts to load real GroundingDINO; raises FileNotFoundError if weights absent.
    """
    if detector is not None:
        return detector.extract_tracklets(frame_paths, text_prompt)

    # Real GroundingDINO loading
    ckpt_path = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    config_path = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"GroundingDINO checkpoint not found at '{ckpt_path}'. "
            "Please download weights or provide an injected detector."
        )

    try:
        import torch
        from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
        from torchvision.ops import box_convert
    except ImportError as e:
        raise RuntimeError(f"Missing required dependencies for GroundingDINO: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config_path, ckpt_path, device=device)

    frame_detections = []
    for fpath in frame_paths:
        _, image = load_image(fpath)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
        if boxes.shape[0] > 0:
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
            scores = logits.cpu().numpy()
            dets = [
                {
                    "box": xyxy[i].tolist(),
                    "score": float(scores[i]),
                    "label": phrases[i] if i < len(phrases) else "object"
                }
                for i in range(len(scores))
            ]
        else:
            dets = []
        frame_detections.append(dets)

    # Greedy IoU matching across frames into tracklets
    return link_detections_into_tracklets(frame_detections)


def link_detections_into_tracklets(frame_detections: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Links per-frame detections across 10 frames into tracklets via greedy spatial IoU.
    """
    def box_iou(b1, b2):
        if len(b1) != 4 or len(b2) != 4 or b1 == [0, 0, 0, 0] or b2 == [0, 0, 0, 0]:
            return 0.0
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / (union + 1e-6) if union > 0 else 0.0

    tracklets = []
    num_frames = len(frame_detections)

    for cand_idx in range(3):
        track_boxes = []
        track_scores = []
        labels = []
        last_box = None

        for t in range(num_frames):
            dets = frame_detections[t]
            if not dets:
                track_boxes.append(last_box if last_box is not None else [0.0, 0.0, 0.0, 0.0])
                track_scores.append(0.0)
                continue

            if last_box is None:
                chosen = dets[min(cand_idx, len(dets) - 1)]
            else:
                ious = [box_iou(last_box, d["box"]) for d in dets]
                best_i = int(np.argmax(ious))
                chosen = dets[best_i] if ious[best_i] > 0.1 else dets[min(cand_idx, len(dets) - 1)]

            track_boxes.append(chosen["box"])
            track_scores.append(chosen.get("score", 0.5))
            labels.append(chosen.get("label", "object"))
            last_box = chosen["box"]

        mean_score = float(np.mean(track_scores)) if track_scores else 0.0
        primary_label = max(set(labels), key=labels.count) if labels else "object"

        # Determine spatial location: left, center, right
        centers_x = [(b[0] + b[2]) / 2.0 for b in track_boxes if b != [0, 0, 0, 0]]
        avg_cx = float(np.mean(centers_x)) if centers_x else 0.5
        if avg_cx < 0.38:
            side = "left"
        elif avg_cx > 0.62:
            side = "right"
        else:
            side = "center"

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in track_boxes if b != [0, 0, 0, 0]]
        mean_area = float(np.mean(areas)) if areas else 0.0

        tracklets.append({
            "cand_id": cand_idx,
            "label": primary_label,
            "boxes": track_boxes,
            "mean_score": round(mean_score, 3),
            "mean_area": round(mean_area, 3),
            "spatial_location": side,
        })

    return tracklets


def extract_audio_events_beats(
    audio_path: str,
    tagger: Optional[Any] = None,
    beats_ckpt: str = "pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
) -> List[Dict[str, Any]]:
    """
    Extracts acoustic events per second using the BEATs model.
    If tagger is None, loads real BEATs from beats_ckpt.
    Raises FileNotFoundError if checkpoint is missing.
    """
    if tagger is not None:
        return tagger.extract_events(audio_path)

    if not os.path.exists(beats_ckpt):
        raise FileNotFoundError(
            f"BEATs acoustic checkpoint not found at '{beats_ckpt}'. "
            "Please download weights to pretrained_weights/ or provide an injected tagger."
        )

    try:
        import torch
        from models.beats.BEATs import BEATs, BEATsConfig
    except ImportError as e:
        raise RuntimeError(f"Missing required dependencies for BEATs: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(beats_ckpt, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    samples, sr = read_audio_waveform(audio_path)
    loudness_curve = compute_audio_loudness_per_second(samples, sr, num_seconds=10)

    events = []
    samples_per_sec = max(1, sr)
    for sec in range(10):
        start = sec * samples_per_sec
        end = min(len(samples), (sec + 1) * samples_per_sec)
        chunk = samples[start:end]
        if len(chunk) > 0:
            with torch.no_grad():
                inp = torch.tensor(chunk, dtype=torch.float32, device=device).unsqueeze(0)
                padding_mask = torch.zeros((1, len(chunk)), dtype=torch.bool, device=device)
                logits = model.extract_features(inp, padding_mask=padding_mask)[0]
                topk = torch.topk(logits.mean(dim=1).squeeze(0), k=3).indices.cpu().numpy()
                top_labels = [get_audioset_label(int(idx)) for idx in topk]
        else:
            top_labels = ["Silence", "Silence", "Silence"]

        events.append({
            "second": sec,
            "interval": [sec, sec + 1],
            "labels": top_labels,
            "primary_label": top_labels[0],
            "loudness": loudness_curve[sec],
        })

    return events


def build_graph(
    video_id: str,
    media_dir: str = "./REFAVS/media",
    kg_dir: str = DEFAULT_KG_DIR,
    reference_expression: Optional[str] = None,
    detector: Optional[Any] = None,
    audio_tagger: Optional[Any] = None,
    meta_csv_path: Optional[str] = None,
    correlation_threshold: float = 0.3,
    loudness_floor: float = 0.05,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    Builds the multimodal Knowledge Graph derived from real video frames and audio:
    - Visual tracklets with pixel-computed motion energy.
    - Audio events per second with real RMS loudness.
    - Pearson-correlation-driven sounding edges.
    - Deterministic entity and relation linking.
    - Validates cache via schema_version and config_hash.
    """
    os.makedirs(kg_dir, exist_ok=True)
    cache_path = os.path.join(kg_dir, f"{video_id}.json")

    config_info = {
        "schema_version": SCHEMA_VERSION,
        "corr_thresh": correlation_threshold,
        "loudness_floor": loudness_floor,
        "ref": reference_expression or "",
    }
    config_hash = compute_config_hash(config_info)

    # Check cache validity
    if not force_recompute and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("schema_version") == SCHEMA_VERSION and cached.get("config_hash") == config_hash:
                return cached
        except Exception:
            pass

    # 1. Resolve media paths (raises FileNotFoundError if missing)
    frame_paths, audio_path = resolve_media_paths(video_id, media_dir)

    # Load 10 frames as PIL images for pixel processing
    frame_images = [Image.open(p).convert("RGB") for p in frame_paths]

    # 2. Visual Entity Extraction (GroundingDINO)
    noun_phrases = extract_noun_phrases(reference_expression or "")
    vocab = get_object_vocabulary(meta_csv_path)
    prompt_tokens = noun_phrases + [v for v in vocab[:10] if v not in noun_phrases]
    text_prompt = " . ".join(prompt_tokens) + " ." if prompt_tokens else "object ."

    tracklets = run_grounding_dino_on_frames(
        frame_paths=frame_paths,
        text_prompt=text_prompt,
        detector=detector,
    )

    # 3. Compute pixel-level motion energy for each visual tracklet
    visual_entities = []
    for trk in tracklets:
        boxes = trk["boxes"]
        motion_curve = compute_tracklet_motion_energy(frame_images, boxes)
        mean_motion = float(np.mean(motion_curve)) if motion_curve else 0.0

        v_node = {
            "id": f"{video_id}_vis_{trk['cand_id']}",
            "type": "visual_entity",
            "cand_id": trk["cand_id"],
            "label": trk["label"],
            "spatial_location": trk["spatial_location"],
            "boxes": boxes,
            "mean_score": trk["mean_score"],
            "area_ratio": trk["mean_area"],
            "motion_energy_curve": motion_curve,
            "motion_energy": round(mean_motion, 4),
        }
        visual_entities.append(v_node)

    # 4. Audio Event Extraction (BEATs + Real RMS Loudness)
    samples, sr = read_audio_waveform(audio_path)
    loudness_curve = compute_audio_loudness_per_second(samples, sr, num_seconds=10)

    if audio_tagger is not None:
        raw_events = audio_tagger.extract_events(audio_path)
    else:
        # Check BEATs weights availability
        beats_ckpt = "pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        if not os.path.exists(beats_ckpt):
            raise FileNotFoundError(
                f"BEATs acoustic checkpoint not found at '{beats_ckpt}'. "
                "Download checkpoint or provide an injected audio_tagger."
            )
        raw_events = extract_audio_events_beats(audio_path, tagger=None, beats_ckpt=beats_ckpt)

    audio_events = []
    for ev in raw_events:
        sec = ev.get("second", 0)
        loudness_val = loudness_curve[sec] if sec < len(loudness_curve) else 0.0
        a_node = {
            "id": f"{video_id}_aud_sec{sec}",
            "type": "audio_event",
            "second": sec,
            "interval": ev.get("interval", [sec, sec + 1]),
            "label": ev.get("primary_label", "Acoustic_Sound"),
            "labels": ev.get("labels", [ev.get("primary_label", "Sound")]),
            "loudness": loudness_val,
        }
        audio_events.append(a_node)

    # 5. Assemble Graph Nodes and Edges
    nodes = {}
    edges = []

    for v in visual_entities:
        nodes[v["id"]] = v

    for a in audio_events:
        nodes[a["id"]] = a

    # Spatial edges between visual entities
    for i in range(len(visual_entities)):
        for j in range(i + 1, len(visual_entities)):
            v1, v2 = visual_entities[i], visual_entities[j]
            loc1, loc2 = v1["spatial_location"], v2["spatial_location"]
            if loc1 == "left" and loc2 != "left":
                edges.append({"source": v1["id"], "target": v2["id"], "relation": "left_of", "weight": 0.85})
            elif loc1 == "right" and loc2 != "right":
                edges.append({"source": v1["id"], "target": v2["id"], "relation": "right_of", "weight": 0.85})
            if v1["area_ratio"] > v2["area_ratio"] * 1.5:
                edges.append({"source": v1["id"], "target": v2["id"], "relation": "in_front_of", "weight": 0.75})

    # Sounding edges via Pearson correlation between entity motion and audio loudness
    loudness_series = np.array([a["loudness"] for a in audio_events], dtype=np.float32)
    mean_loudness = float(np.mean(loudness_series)) if len(loudness_series) > 0 else 0.0

    for v in visual_entities:
        motion_series = np.array(v.get("motion_energy_curve", []), dtype=np.float32)
        if len(motion_series) == len(loudness_series) and len(motion_series) > 2:
            std_m = float(np.std(motion_series))
            std_l = float(np.std(loudness_series))
            if std_m > 1e-4 and std_l > 1e-4:
                corr = float(np.corrcoef(motion_series, loudness_series)[0, 1])
            else:
                corr = 0.0

            if corr >= correlation_threshold and mean_loudness >= loudness_floor:
                # Link sounding edge to active audio event seconds
                for a in audio_events:
                    if a["loudness"] >= loudness_floor:
                        edges.append({
                            "source": v["id"],
                            "target": a["id"],
                            "relation": "sounding",
                            "weight": round(corr, 3),
                        })

    # Temporal ordering edges between audio events
    for i in range(len(audio_events) - 1):
        edges.append({
            "source": audio_events[i]["id"],
            "target": audio_events[i + 1]["id"],
            "relation": "temporal_next",
            "weight": 1.0,
        })

    # Referring Expression linking (refers_to) based on label semantic match
    if reference_expression and visual_entities:
        text_id = f"{video_id}_text_query"
        nodes[text_id] = {
            "id": text_id,
            "type": "text_query",
            "text": reference_expression,
        }

        # Match referring expression to best tracklet label
        ref_words = set(re.findall(r"\b\w+\b", reference_expression.lower()))
        best_cand = visual_entities[0]
        best_match_score = -1

        for v in visual_entities:
            v_words = set(re.findall(r"\b\w+\b", v["label"].lower()))
            overlap = len(ref_words & v_words)
            if overlap > best_match_score:
                best_match_score = overlap
                best_cand = v

        edges.append({
            "source": text_id,
            "target": best_cand["id"],
            "relation": "refers_to",
            "weight": 0.90 if best_match_score > 0 else 0.50,
        })

    graph_data = {
        "video_id": video_id,
        "schema_version": SCHEMA_VERSION,
        "config_hash": config_hash,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    return graph_data


build_and_save_kg = build_graph


def main():
    parser = argparse.ArgumentParser(description="Extract and cache media-driven multimodal KG")
    parser.add_argument("--video_id", type=str, required=True, help="Video UID")
    parser.add_argument("--media_dir", type=str, default="./REFAVS/media", help="Media directory")
    parser.add_argument("--kg_dir", type=str, default=DEFAULT_KG_DIR, help="Cache directory")
    parser.add_argument("--ref", type=str, default=None, help="Referring expression")
    args = parser.parse_args()

    g = build_graph(args.video_id, media_dir=args.media_dir, kg_dir=args.kg_dir, reference_expression=args.ref)
    print(f"Generated KG for {args.video_id}: {g['num_nodes']} nodes, {g['num_edges']} edges.")


if __name__ == "__main__":
    main()
