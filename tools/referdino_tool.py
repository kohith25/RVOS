from __future__ import annotations
#!/usr/bin/env python3
"""
tools/referdino_tool.py - ReferDINO / GroundingDINO Grounding Tools

Implements:
1. ReferDINOTool: Real ReferDINO (ICCV 2025: Referring Video Object Segmentation with
   Visual Grounding Foundations). Operates over 10 video frames as a clip using the
   full referring sentence, returning spatio-temporal track candidates.
   Strictly separated from GroundingDINO: zero GroundingDINO imports or weights used here.
2. GroundingDINOTool: GroundingDINO baseline tool that links frame-level bounding boxes
   across consecutive frames via spatial IoU tracking.
3. Candidate schema: cand_id, boxes (10 frames), masks (optional), score, mean_area, side.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from tools.mock_tools import MockCandidate, MockReferDINOTool


@dataclass
class Candidate:
    cand_id: int
    boxes: List[List[float]]
    masks: Optional[List[Any]] = None
    score: float = 0.0
    mean_area: float = 0.0
    side: str = "center"

    def to_dict(self):
        return asdict(self)


def compute_side_from_boxes(boxes: List[List[float]]) -> str:
    center_xs = [((b[0] + b[2]) / 2.0) for b in boxes if len(b) == 4 and b != [0, 0, 0, 0]]
    if not center_xs:
        return "center"
    mean_cx = float(np.mean(center_xs))
    if mean_cx < 0.38:
        return "left"
    elif mean_cx > 0.62:
        return "right"
    return "center"


def compute_mean_area_from_boxes(boxes: List[List[float]]) -> float:
    areas = [max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1])) for b in boxes if len(b) == 4]
    return float(np.mean(areas)) if areas else 0.0


class ReferDINOTool:
    """
    Adapter for the official ReferDINO ``demo_video.py`` inference path.

    The upstream demo selects one query track for a sentence. This adapter returns
    that track as candidate 0; it does not synthesize additional candidates.
    """
    def __init__(
        self,
        config_path: str = "configs/ytvos_swinb.yaml",
        checkpoint_path: str = "ckpt/ryt_mevis_swinb.pth",
        referdino_repo: str = "third_party/ReferDINO",
        device: Optional[str] = None,
        media_dir: str = "./REFAVS/media",
        score_threshold: float = 0.20,
        use_mock: bool = False,
        model: Optional[Any] = None,
    ):
        self.repo_dir = os.path.abspath(referdino_repo)
        self.config_path = self._resolve_upstream_path(config_path)
        self.checkpoint_path = self._resolve_upstream_path(checkpoint_path)
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.media_dir = media_dir
        self.score_threshold = score_threshold
        self.use_mock = use_mock
        self.model = model
        self._runtime = None
        self._args = None

        if self.use_mock:
            self._mock_impl = MockReferDINOTool()
        else:
            self._mock_impl = None
            if self.model is None:
                self._load_model()

    def _resolve_upstream_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_dir, path)

    def _load_model(self):
        """Load the model using the builder and checkpoint format used upstream."""
        if not os.path.isdir(self.repo_dir):
            raise FileNotFoundError(
                f"ReferDINO repository not found at '{self.repo_dir}'. "
                "Clone https://github.com/iSEE-Laboratory/ReferDINO to "
                "third_party/ReferDINO and follow its README setup steps "
                "documented in RLGRPO.md."
            )
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(
                f"ReferDINO config not found at '{self.config_path}'. "
                "Use the upstream configs/ytvos_swinb.yaml from "
                "https://github.com/iSEE-Laboratory/ReferDINO."
            )
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"ReferDINO checkpoint not found at '{self.checkpoint_path}'. "
                "Download ckpt/ryt_mevis_swinb.pth from "
                "https://huggingface.co/liangtm/referdino/tree/main, or set "
                "checkpoint_path to another checkpoint listed in the upstream README."
            )
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for ReferDINO. Follow the ReferDINO README "
                "environment setup in RLGRPO.md."
            )

        repo_prefix = os.path.normcase(os.path.realpath(self.repo_dir)) + os.sep
        loaded_models = sys.modules.get("models")
        loaded_path = getattr(loaded_models, "__file__", None)
        if loaded_path and not os.path.normcase(os.path.realpath(loaded_path)).startswith(repo_prefix):
            raise ImportError(
                "ReferDINO and TGS-Agent both use the top-level package name 'models'. "
                "Run ReferDINO in the separate tool-server process/environment described "
                "in RLGRPO.md so its upstream models package can be imported without conflict."
            )
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        try:
            try:
                from easydict import EasyDict
                from ruamel.yaml import YAML
                import torch.nn.functional as functional
                import torchvision.transforms as transforms
                from torch.cuda.amp import autocast
                from misc import nested_tensor_from_videos_list
                from models import build_model
            except ImportError as exc:
                raise ImportError(
                    "Could not import ReferDINO's upstream inference dependencies. "
                    "Install dependencies and build its custom attention ops as described "
                    "in the ReferDINO README and RLGRPO.md."
                ) from exc

            with open(self.config_path, "r", encoding="utf-8") as config_file:
                config = YAML(typ="safe", pure=True).load(config_file)
            config = {key: value["value"] for key, value in config.items()}
            config.update({
                "checkpoint_path": self.checkpoint_path,
                "device": self.device,
                "frame_step": 1,
                "tracking_alpha": 0.1,
            })
            self._args = EasyDict(config)
            self._args.GroundingDINO.tracking_alpha = self._args.tracking_alpha

            self.model, _, _ = build_model(self._args)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.model.to(self.device)
            self.model.eval()
            self._runtime = {
                "functional": functional,
                "transforms": transforms,
                "autocast": autocast,
                "nested_tensor_from_videos_list": nested_tensor_from_videos_list,
            }
        finally:
            for name, module in list(sys.modules.items()):
                module_path = getattr(module, "__file__", None)
                if module_path and os.path.normcase(os.path.realpath(module_path)).startswith(repo_prefix):
                    sys.modules.pop(name, None)
            if self.repo_dir in sys.path:
                sys.path.remove(self.repo_dir)

    def _resolve_clip_frames(self, video_id: str) -> List[str]:
        vid = video_id.rsplit("_", 2)[0] if "_" in video_id else video_id
        candidates = [
            os.path.join(self.media_dir, vid, "frames"),
            os.path.join(self.media_dir, video_id, "frames"),
            os.path.join(self.media_dir, vid),
        ]
        frame_dir = None
        for c in candidates:
            if os.path.isdir(c):
                frame_dir = c
                break

        if frame_dir is None:
            raise FileNotFoundError(f"Video frame directory not found for '{video_id}' in '{self.media_dir}'.")

        paths = []
        for i in range(10):
            for ext in [".jpg", ".png", ".jpeg"]:
                p = os.path.join(frame_dir, f"{i}{ext}")
                if os.path.exists(p):
                    paths.append(p)
                    break
        if len(paths) < 10:
            all_imgs = sorted([
                os.path.join(frame_dir, f) for f in os.listdir(frame_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])
            if len(all_imgs) >= 10:
                paths = all_imgs[:10]

        if len(paths) < 10:
            raise FileNotFoundError(f"Expected 10 frames for video '{video_id}', found {len(paths)}.")
        return paths

    def run(self, video_id: str, text: str) -> List[Candidate]:
        """
        Runs spatio-temporal video referring grounding over the 10-frame clip.
        Returns a list of Candidate tracklets sorted by confidence score.
        """
        if self.use_mock:
            mock_cands = self._mock_impl.run(video_id, text)
            return [
                Candidate(
                    cand_id=c.cand_id,
                    boxes=c.boxes,
                    masks=c.masks,
                    score=c.score,
                    mean_area=c.mean_area,
                    side=c.side,
                )
                for c in mock_cands
            ]

        if self.model is None:
            raise RuntimeError("ReferDINO model is not initialized.")

        frame_paths = self._resolve_clip_frames(video_id)

        # Injected stub path for unit testing
        if hasattr(self.model, "predict_clip"):
            raw_tracks = self.model.predict_clip(frame_paths, text)
            candidates = []
            for idx, trk in enumerate(raw_tracks):
                boxes = trk.get("boxes", [[0, 0, 0, 0]] * 10)
                masks = trk.get("masks", None)
                score = trk.get("score", 0.8)
                candidates.append(Candidate(
                    cand_id=idx,
                    boxes=boxes,
                    masks=masks,
                    score=round(score, 3),
                    mean_area=round(compute_mean_area_from_boxes(boxes), 3),
                    side=compute_side_from_boxes(boxes),
                ))
            candidates.sort(key=lambda c: c.score, reverse=True)
            for i, c in enumerate(candidates):
                c.cand_id = i
            return candidates

        # Match the preprocessing and model.infer API in the upstream demo_video.py.
        frames_pil = [Image.open(p).convert("RGB") for p in frame_paths]
        if self._runtime is None:
            self._load_model()
        transforms = self._runtime["transforms"]
        functional = self._runtime["functional"]
        transform = transforms.Compose([
            transforms.Resize(360),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_sizes = [frame.size for frame in frames_pil]
        imgs = torch.stack([transform(frame) for frame in frames_pil], dim=0).to(self.device)
        samples = self._runtime["nested_tensor_from_videos_list"](
            imgs[None], size_divisibility=1
        )
        img_h, img_w = imgs.shape[-2:]
        target = {"size": torch.as_tensor([int(img_h), int(img_w)], device=self.device)}
        sentence = " ".join(text.lower().split())
        with torch.no_grad():
            with self._runtime["autocast"](self._args.enable_amp):
                outputs = self.model.infer(samples, [sentence], [target])

        pred_logits = outputs["pred_logits"][0]
        pred_masks = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]
        query_scores = pred_logits.sigmoid().mean(0).max(-1).values
        query_id = int(query_scores.argmax().item())
        score = float(query_scores[query_id].item())
        frame_indices = torch.arange(len(frames_pil), device=pred_boxes.device)

        cxcywh = pred_boxes[frame_indices, query_id].detach().cpu().numpy()
        boxes = np.stack([
            cxcywh[:, 0] - cxcywh[:, 2] / 2.0,
            cxcywh[:, 1] - cxcywh[:, 3] / 2.0,
            cxcywh[:, 0] + cxcywh[:, 2] / 2.0,
            cxcywh[:, 1] + cxcywh[:, 3] / 2.0,
        ], axis=1)
        boxes = np.clip(boxes, 0.0, 1.0).tolist()

        mask_logits = pred_masks[frame_indices, query_id, :, :]
        mask_logits = mask_logits[:, :img_h, :img_w]
        masks = []
        for frame_idx, (origin_w, origin_h) in enumerate(image_sizes):
            frame_mask = functional.interpolate(
                mask_logits[frame_idx].float()[None, None],
                size=(origin_h, origin_w),
                mode="bilinear",
                align_corners=False,
            )
            masks.append((frame_mask.sigmoid()[0, 0] > 0.5).to(torch.uint8).cpu().tolist())

        return [Candidate(
            cand_id=0,
            boxes=boxes,
            masks=masks,
            score=round(score, 3),
            mean_area=round(compute_mean_area_from_boxes(boxes), 3),
            side=compute_side_from_boxes(boxes),
        )]


class GroundingDINOTool:
    """
    GroundingDINO baseline tool that links frame-level bounding boxes across
    consecutive video frames via spatial IoU tracking.
    """
    def __init__(
        self,
        config_path: str = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "gdino_checkpoints/groundingdino_swint_ogc.pth",
        device: Optional[str] = None,
        media_dir: str = "./REFAVS/media",
        box_threshold: float = 0.15,
        text_threshold: float = 0.25,
        use_mock: bool = False,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
        self.media_dir = media_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.use_mock = use_mock
        self.model = None

        if self.use_mock:
            self._mock_impl = MockReferDINOTool()
        else:
            self._mock_impl = None
            self._load_model()

    def _load_model(self):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GroundingDINO inference.")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found at: {self.checkpoint_path}. "
                "Ensure weights are placed in gdino_checkpoints/ or pass use_mock=True."
            )
        from grounding_dino.groundingdino.util.inference import load_model
        self.model = load_model(
            model_config_path=self.config_path,
            model_checkpoint_path=self.checkpoint_path,
            device=self.device
        )

    def run(self, video_id: str, text: str) -> List[Candidate]:
        if self.use_mock:
            mock_cands = self._mock_impl.run(video_id, text)
            return [
                Candidate(
                    cand_id=c.cand_id,
                    boxes=c.boxes,
                    masks=c.masks,
                    score=c.score,
                    mean_area=c.mean_area,
                    side=c.side,
                )
                for c in mock_cands
            ]

        if self.model is None:
            raise RuntimeError("GroundingDINO model is not loaded.")

        from grounding_dino.groundingdino.util.inference import load_image, predict
        try:
            from torchvision.ops import box_convert
        except ImportError:
            box_convert = None

        vid = video_id.rsplit("_", 2)[0] if "_" in video_id else video_id
        frame_dir = os.path.join(self.media_dir, vid, "frames")
        if not os.path.exists(frame_dir):
            raise FileNotFoundError(f"Video frames directory not found: {frame_dir}")

        frame_detections = []
        for img_id in range(10):
            img_path = os.path.join(frame_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                frame_detections.append([])
                continue

            _, image = load_image(img_path)
            boxes, logits, _ = predict(
                model=self.model,
                image=image,
                caption=text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            if boxes.shape[0] > 0 and box_convert is not None:
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
                scores = logits.cpu().numpy()
                frame_dets = [{"box": xyxy[i].tolist(), "score": float(scores[i])} for i in range(len(scores))]
            else:
                frame_dets = []
            frame_detections.append(frame_dets)

        # IoU tracking across 10 frames
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
        for cand_idx in range(3):
            track_boxes = []
            track_scores = []
            last_box = None

            for t in range(10):
                dets = frame_detections[t]
                if not dets:
                    track_boxes.append(last_box if last_box else [0.0, 0.0, 0.0, 0.0])
                    track_scores.append(0.0)
                    continue

                if last_box is None:
                    chosen_det = dets[min(cand_idx, len(dets) - 1)]
                else:
                    ious = [box_iou(last_box, d["box"]) for d in dets]
                    best_match = int(np.argmax(ious))
                    chosen_det = dets[best_match] if ious[best_match] > 0.1 else dets[min(cand_idx, len(dets) - 1)]

                last_box = chosen_det["box"]
                track_boxes.append(chosen_det["box"])
                track_scores.append(chosen_det["score"])

            mean_score = sum(track_scores) / len(track_scores) if track_scores else 0.0
            tracklets.append(Candidate(
                cand_id=cand_idx,
                boxes=track_boxes,
                masks=None,
                score=round(mean_score, 3),
                mean_area=round(compute_mean_area_from_boxes(track_boxes), 3),
                side=compute_side_from_boxes(track_boxes),
            ))

        tracklets.sort(key=lambda c: c.score, reverse=True)
        for i, c in enumerate(tracklets):
            c.cand_id = i
        return tracklets


_default_referdino = None
_default_gdino = None


def get_referdino_tool(use_mock: bool = False, media_dir: str = "./REFAVS/media") -> ReferDINOTool:
    global _default_referdino
    if _default_referdino is None:
        _default_referdino = ReferDINOTool(use_mock=use_mock, media_dir=media_dir)
    return _default_referdino


def get_grounding_tool(grounder: str = "gdino", use_mock: bool = False, media_dir: str = "./REFAVS/media") -> Any:
    global _default_referdino, _default_gdino
    if grounder == "referdino":
        return get_referdino_tool(use_mock=use_mock, media_dir=media_dir)
    if _default_gdino is None:
        _default_gdino = GroundingDINOTool(use_mock=use_mock, media_dir=media_dir)
    return _default_gdino
