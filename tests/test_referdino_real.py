import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERDINO_REPO = Path(
    os.environ.get("REFERDINO_REPO", REPO_ROOT / "third_party" / "ReferDINO")
)
REFERDINO_CHECKPOINT = Path(
    os.environ.get(
        "REFERDINO_CHECKPOINT",
        REFERDINO_REPO / "ckpt" / "ryt_mevis_swinb.pth",
    )
)
REFERDINO_CONFIG = Path(
    os.environ.get(
        "REFERDINO_CONFIG",
        REFERDINO_REPO / "configs" / "ytvos_swinb.yaml",
    )
)


def test_real_referdino_inference_returns_official_single_query_track():
    if not REFERDINO_REPO.is_dir() or not REFERDINO_CHECKPOINT.is_file():
        pytest.skip(
            "Real ReferDINO checkout/checkpoint unavailable; install it as described in RLGRPO.md."
        )
    video_id = os.environ.get("REFERDINO_VIDEO_ID")
    media_dir = os.environ.get("REFERDINO_MEDIA_DIR", str(REPO_ROOT / "REFAVS" / "media"))
    if not video_id:
        pytest.skip("Set REFERDINO_VIDEO_ID and REFERDINO_MEDIA_DIR for real clip inference.")

    from tools.referdino_tool import ReferDINOTool

    tool = ReferDINOTool(
        referdino_repo=str(REFERDINO_REPO),
        config_path=str(REFERDINO_CONFIG),
        checkpoint_path=str(REFERDINO_CHECKPOINT),
        media_dir=media_dir,
    )
    candidates = tool.run(video_id, "the person playing guitar")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.cand_id == 0
    assert len(candidate.boxes) == 10
    assert candidate.masks is not None and len(candidate.masks) == 10
    assert 0.0 <= candidate.score <= 1.0
    for box in candidate.boxes:
        assert len(box) == 4
        assert 0.0 <= box[0] <= box[2] <= 1.0
        assert 0.0 <= box[1] <= box[3] <= 1.0


def test_no_old_referdino_repository_url_in_project_sources():
    forbidden_url = "github.com/" + "referdino"
    text_suffixes = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".sh"}
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in text_suffixes:
            continue
        if "Grounded-SAM-2" in path.parts:
            continue
        assert forbidden_url not in path.read_text(encoding="utf-8", errors="ignore"), str(path)
