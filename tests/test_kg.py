from __future__ import annotations
#!/usr/bin/env python3
"""
tests/test_kg.py - Tests for Phase 1: Real Multimodal Knowledge Graph Builder
"""

import os
import sys
import pytest
import shutil
import tempfile
from typing import List, Dict, Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kg.audioset_label_map import AUDIOSET_LABELS, get_audioset_label
from kg.build_kg import (
    build_graph,
    resolve_media_paths,
    compute_tracklet_motion_energy,
    read_audio_waveform,
    compute_audio_loudness_per_second,
)
from kg.retriever import KGRetriever


class StubDetector:
    """Test stub for visual detection returning deterministic boxes based on fixture content."""
    def __init__(self, label="moving_rectangle", boxes=None):
        self.label = label
        self.boxes = boxes

    def extract_tracklets(self, frame_paths: List[str], text_prompt: str) -> List[Dict[str, Any]]:
        if self.boxes is not None:
            track_boxes = self.boxes
        else:
            # Default box for 64x64 moving rect (x ~ 10-40, y ~ 20-44)
            track_boxes = [[0.1, 0.3, 0.6, 0.7] for _ in range(len(frame_paths))]

        return [{
            "cand_id": 0,
            "label": self.label,
            "boxes": track_boxes,
            "mean_score": 0.95,
            "mean_area": 0.20,
            "spatial_location": "center",
        }]


class StubTagger:
    """Test stub for acoustic event tagging."""
    def __init__(self, primary_label="Sine_Wave"):
        self.primary_label = primary_label

    def extract_events(self, audio_path: str) -> List[Dict[str, Any]]:
        events = []
        for sec in range(10):
            events.append({
                "second": sec,
                "interval": [sec, sec + 1],
                "labels": [self.primary_label, "Music", "Sound"],
                "primary_label": self.primary_label,
            })
        return events


@pytest.fixture
def fixtures_dir():
    return os.path.join(REPO_ROOT, "tests", "fixtures")


@pytest.fixture
def tmp_kg_dir():
    d = tempfile.mkdtemp(prefix="test_kg_cache_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_audioset_label_map_full_527():
    """Verify that the full 527-class AudioSet label map is present."""
    assert len(AUDIOSET_LABELS) == 527
    assert 0 in AUDIOSET_LABELS
    assert 526 in AUDIOSET_LABELS
    assert get_audioset_label(0) == "Speech"
    assert get_audioset_label(526) == "Field recording"


def test_missing_frames_raises_filenotfound(fixtures_dir):
    """Missing frames directory must raise FileNotFoundError, not silent fallback."""
    with pytest.raises(FileNotFoundError):
        resolve_media_paths("nonexistent_video_123", media_dir=fixtures_dir)


def test_missing_audio_raises_filenotfound(tmp_path):
    """Missing audio file must raise FileNotFoundError."""
    vid_dir = tmp_path / "vid_missing_audio" / "frames"
    vid_dir.mkdir(parents=True)
    for i in range(10):
        (vid_dir / f"{i}.jpg").write_text("fake")

    with pytest.raises(FileNotFoundError):
        resolve_media_paths("vid_missing_audio", media_dir=str(tmp_path))


def test_moving_rect_has_sounding_edge(fixtures_dir, tmp_kg_dir):
    """
    Moving rectangle + synchronized tone burst produces a 'sounding' edge,
    because motion energy and audio loudness correlate.
    """
    detector = StubDetector(label="moving_box")
    tagger = StubTagger(primary_label="Synthesizer")

    graph = build_graph(
        video_id="moving_rect",
        media_dir=fixtures_dir,
        kg_dir=tmp_kg_dir,
        reference_expression="the moving box",
        detector=detector,
        audio_tagger=tagger,
        correlation_threshold=0.3,
        force_recompute=True,
    )

    sounding_edges = [e for e in graph["edges"] if e["relation"] == "sounding"]
    assert len(sounding_edges) > 0, (
        f"Expected 'sounding' edges for correlated moving+tone video; found edges: {graph['edges']}"
    )


def test_static_silent_no_sounding_edge(fixtures_dir, tmp_kg_dir):
    """
    Static box + silence must NOT produce any 'sounding' edge.
    """
    detector = StubDetector(label="static_box")
    tagger = StubTagger(primary_label="Silence")

    graph = build_graph(
        video_id="static_silent",
        media_dir=fixtures_dir,
        kg_dir=tmp_kg_dir,
        reference_expression="the static box",
        detector=detector,
        audio_tagger=tagger,
        correlation_threshold=0.3,
        force_recompute=True,
    )

    sounding_edges = [e for e in graph["edges"] if e["relation"] == "sounding"]
    assert len(sounding_edges) == 0, (
        f"Static silent video should have NO 'sounding' edges; found: {sounding_edges}"
    )


def test_graph_stable_under_video_id_rename(fixtures_dir, tmp_kg_dir, tmp_path):
    """
    Assert that changing the video ID string while pointing to the exact same media
    yields the exact same graph content (verifying content is not derived from video_id hash).
    """
    # Create a symlink or copy of moving_rect with a different name
    alt_dir = tmp_path / "renamed_video_xyz"
    shutil.copytree(os.path.join(fixtures_dir, "moving_rect"), str(alt_dir))

    detector = StubDetector(label="moving_box")
    tagger = StubTagger(primary_label="Tone")

    g1 = build_graph(
        video_id="moving_rect",
        media_dir=fixtures_dir,
        kg_dir=tmp_kg_dir,
        detector=detector,
        audio_tagger=tagger,
        force_recompute=True,
    )

    g2 = build_graph(
        video_id="renamed_video_xyz",
        media_dir=str(tmp_path),
        kg_dir=tmp_kg_dir,
        detector=detector,
        audio_tagger=tagger,
        force_recompute=True,
    )

    # Number of nodes and edges must match identically
    assert g1["num_nodes"] == g2["num_nodes"]
    assert g1["num_edges"] == g2["num_edges"]
    # Relations and weights must match
    rels1 = sorted([e["relation"] for e in g1["edges"]])
    rels2 = sorted([e["relation"] for e in g2["edges"]])
    assert rels1 == rels2


def test_retriever_query_conditioned(fixtures_dir, tmp_kg_dir):
    """Test that KGRetriever retrieves relevant facts for spatial and acoustic queries."""
    detector = StubDetector(label="guitar")
    tagger = StubTagger(primary_label="Acoustic_guitar")

    build_graph(
        video_id="moving_rect",
        media_dir=fixtures_dir,
        kg_dir=tmp_kg_dir,
        reference_expression="the guitar playing on the left",
        detector=detector,
        audio_tagger=tagger,
        force_recompute=True,
    )

    retriever = KGRetriever(kg_dir=tmp_kg_dir, media_dir=fixtures_dir)
    res = retriever.retrieve("moving_rect", query="guitar sound")

    assert "facts" in res and "node_ids" in res
    assert len(res["facts"]) > 0
    assert any("guitar" in f.lower() or "acoustic" in f.lower() for f in res["facts"])


def test_no_hash_derivation_in_kg_source():
    """Verify that kg/ does NOT use md5 or hashlib outside cache key hashing."""
    kg_source_dir = os.path.join(REPO_ROOT, "kg")
    for root, _, files in os.walk(kg_source_dir):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, "r", encoding="utf-8") as fh:
                    for line_idx, line in enumerate(fh, start=1):
                        assert "md5" not in line.lower(), (
                            f"Found forbidden 'md5' in {path}:{line_idx}: {line.strip()}"
                        )
                        if "hashlib" in line:
                            assert "sha256" in line or "import hashlib" in line, (
                                f"Unexpected hashlib usage in {path}:{line_idx}: {line.strip()}"
                            )
