from __future__ import annotations
#!/usr/bin/env python3
"""
scripts/smoke_referdino.py - Smoke test script for ReferDINO inference on 1 video clip.

Usage:
    python scripts/smoke_referdino.py --video_dir ./REFAVS/media/vid_001 --expr "the person playing guitar"
"""

import os
import sys
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.referdino_tool import ReferDINOTool


def main():
    parser = argparse.ArgumentParser(description="Smoke test for ReferDINO on a single video clip")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to video directory containing frames/ or 0..9.jpg")
    parser.add_argument("--expr", type=str, required=True, help="Natural language referring expression")
    parser.add_argument("--repo_dir", type=str, default="third_party/ReferDINO", help="ReferDINO repository checkout")
    parser.add_argument("--ckpt", type=str, default="ckpt/ryt_mevis_swinb.pth", help="Path relative to ReferDINO repo or absolute")
    parser.add_argument("--config", type=str, default="configs/ytvos_swinb.yaml", help="Path relative to ReferDINO repo or absolute")
    parser.add_argument("--mock", action="store_true", help="Run with mock implementation on CPU")
    args = parser.parse_args()

    media_dir = os.path.dirname(os.path.abspath(args.video_dir))
    video_id = os.path.basename(os.path.abspath(args.video_dir))

    print(f"[ReferDINO Smoke Test] Video: {video_id} in {media_dir}")
    print(f"[ReferDINO Smoke Test] Expression: '{args.expr}'")
    print(f"[ReferDINO Smoke Test] Mock: {args.mock}")

    tool = ReferDINOTool(
        config_path=args.config,
        checkpoint_path=args.ckpt,
        referdino_repo=args.repo_dir,
        media_dir=media_dir,
        use_mock=args.mock,
    )

    candidates = tool.run(video_id, args.expr)
    print(f"\n[Result] Extracted {len(candidates)} candidate tracks:")
    for c in candidates:
        print({
            "cand_id": c.cand_id,
            "score": c.score,
            "side": c.side,
            "mean_area": c.mean_area,
            "boxes": c.boxes,
            "num_masks": len(c.masks) if c.masks is not None else 0,
        })


if __name__ == "__main__":
    main()
