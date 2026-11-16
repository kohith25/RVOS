#!/usr/bin/env python3
"""
tests/test_tools.py - Comprehensive Unit Tests for Tool Suite, Server, Client & Action Schema

Verifies:
1. Deterministic disk caching in ToolClient.
2. Ground Truth isolation: tools never receive or access GT annotations.
3. Action parser & schema validation: invalid actions / JSON give penalties, never silent stop.
4. ReferDINO tool produces 10-frame candidate tubelets with score, mean_area, and side.
5. SAM2 tool refines bounding boxes into valid (10, 224, 224) binary masks.
6. KG retriever extracts spatial and acoustic relational facts.
7. Observation formatters generate clean conversational context.
"""

import os
import sys
import json
import ast
import tempfile
import inspect
import pytest
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.referdino_tool import ReferDINOTool, Candidate
from tools.sam2_tool import SAM2Tool
from tools.mock_server import MockToolServer
from tools.client import ToolClient
from kg.build_kg import build_graph
from kg.retriever import KGRetriever
from rl_grpo.constants import (
    ALLOWED_ACTIONS,
    parse_and_validate_action,
    format_observation,
    ACTION_TO_COST
)


def test_gt_isolation():
    """Verify that tool methods never accept GT parameters in their signatures."""
    for cls in (ReferDINOTool, SAM2Tool, MockToolServer, ToolClient):
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            sig = inspect.signature(method)
            for param in sig.parameters.keys():
                assert "gt" not in param.lower(), (
                    f"Violation of GT isolation: Method {cls.__name__}.{name} accepts parameter '{param}'!"
                )


def test_action_parser_valid_actions():
    """Verify that all allowed actions with valid arguments pass schema validation."""
    valid_samples = [
        ('{"thought": "retrieve context", "action": "retrieve_evidence", "args": {"query": "woman playing guitar"}}', "retrieve_evidence"),
        ('{"thought": "check frames", "action": "inspect_frames", "args": {"start": 2, "end": 6}}', "inspect_frames"),
        ('{"thought": "refine query", "action": "refine_description", "args": {"text": "acoustic guitar center"}}', "refine_description"),
        ('{"thought": "run grounding", "action": "call_referdino", "args": {"text": "guitar on the left"}}', "call_referdino"),
        ('{"thought": "segment best candidate", "action": "call_sam2", "args": {"candidate_id": 1}}', "call_sam2"),
        ('{"thought": "done", "action": "stop", "args": {}}', "stop"),
    ]
    for raw_text, expected_action in valid_samples:
        action_dict, err = parse_and_validate_action(raw_text)
        assert err is None, f"Unexpected validation error for {raw_text}: {err}"
        assert action_dict["action"] == expected_action


def test_action_parser_invalid_cases():
    """Verify that malformed JSON, unknown actions, and missing arguments are caught and penalized."""
    invalid_samples = [
        ("this is not json at all", "Failed to parse a valid JSON object"),
        ('{"action": "unknown_action", "args": {}}', "is not in allowed actions"),
        ('{"action": "call_sam2", "args": {}}', "requires integer argument 'candidate_id'"),
        ('{"action": "inspect_frames", "args": {"start": 8, "end": 2}}', "Frame range [8, 2] must be within [0, 9] with start <= end"),
        ('{"action": "retrieve_evidence", "args": {"query": ""}}', "requires non-empty string argument 'query'"),
        ('{"thought": "no action specified"}', "Missing or invalid 'action' key in JSON"),
    ]
    for raw_text, expected_err_substr in invalid_samples:
        action_dict, err = parse_and_validate_action(raw_text)
        assert action_dict is None, f"Expected failure for {raw_text}, but got valid dict: {action_dict}"
        assert expected_err_substr in err, f"Expected '{expected_err_substr}' in error, got '{err}'"


def test_referdino_tool():
    """Verify ReferDINO candidate generation."""
    tool = ReferDINOTool(use_mock=True)
    candidates = tool.run("test_vid_01", "man singing in the center")

    assert len(candidates) >= 3
    for c in candidates:
        assert isinstance(c, Candidate)
        assert len(c.boxes) == 10
        for box in c.boxes:
            assert len(box) == 4
            assert 0.0 <= box[0] <= box[2] <= 1.0
            assert 0.0 <= box[1] <= box[3] <= 1.0
        assert 0.0 <= c.score <= 1.0
        assert c.side in ("left", "center", "right")


def test_sam2_tool():
    """Verify SAM2 mask and logit refinement."""
    tool = SAM2Tool(use_mock=True)
    dummy_boxes = [[0.2, 0.2, 0.6, 0.6]] * 10
    masks, logits = tool.refine("test_vid_01", dummy_boxes)

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (10, 224, 224)
    assert masks.dtype == np.uint8
    assert masks.sum() > 0

    assert isinstance(logits, np.ndarray)
    assert logits.shape == (10, 224, 224)
    assert logits.dtype == np.float32


def test_kg_retriever():
    """Verify Multimodal Knowledge Graph construction and retrieval."""
    fixtures_dir = os.path.join(REPO_ROOT, "tests", "fixtures")
    from tests.test_kg import StubDetector, StubTagger
    with tempfile.TemporaryDirectory() as kg_dir:
        build_graph(
            "moving_rect",
            media_dir=fixtures_dir,
            kg_dir=kg_dir,
            reference_expression="the guitar playing on the left",
            detector=StubDetector(label="guitar"),
            audio_tagger=StubTagger(primary_label="Acoustic_guitar"),
        )
        res = KGRetriever(kg_dir=kg_dir, media_dir=fixtures_dir).retrieve(
            "moving_rect", "the guitar playing on the left", k_hops=1
        )
    assert "facts" in res
    assert "node_ids" in res
    assert len(res["facts"]) > 0
    facts_str = " ".join(res["facts"]).lower()
    assert any(term in facts_str for term in ("entity", "acoustic", "left", "sound", "activity"))


def test_tool_client_caching():
    """Verify that ToolClient disk caching is deterministic and functional."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "cache")
        client = ToolClient(cache_dir=cache_dir, use_mock_fallback=True)

        # Call 1: uncached
        cands1 = client.referdino_ground("vid_cache_test", "guitar player")
        assert len(cands1) > 0

        # Verify cache file was written
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) == 1

        # Call 2: should hit cache
        cands2 = client.referdino_ground("vid_cache_test", "guitar player")
        assert cands1 == cands2


def test_observation_formatter():
    """Verify that observation formatting produces readable agent prompts."""
    obs_kg = format_observation("retrieve_evidence", {"facts": ["Entity guitar is on the left.", "Entity man is center."]})
    assert "Retrieved Knowledge Graph Evidence:" in obs_kg
    assert "Entity guitar is on the left." in obs_kg

    obs_dino = format_observation("call_referdino", {"candidates": [{"cand_id": 0, "side": "left", "score": 0.85, "mean_area": 0.20}]})
    assert "ReferDINO generated candidates:" in obs_dino
    assert "Candidate 0: location=left" in obs_dino


def test_client_unreachable_raises_without_mock():
    """FIX 5(c): When mock fallback is disabled, an unreachable server must raise ConnectionError."""
    # Use unused port on localhost
    client = ToolClient(server_url="http://127.0.0.1:59999", use_mock_fallback=False, use_cache=False)
    with pytest.raises(ConnectionError, match="Tool server not reachable at http://127.0.0.1:59999"):
        client.referdino_ground("v1", "dog barking")


def test_kg_different_videos_produce_distinct_graphs():
    """Two distinct videos must produce different knowledge graphs derived from real media."""
    fixtures_dir = os.path.join(REPO_ROOT, "tests", "fixtures")
    from tests.test_kg import StubDetector, StubTagger
    detector = StubDetector()
    tagger = StubTagger()

    with tempfile.TemporaryDirectory() as tmp_dir:
        g1 = build_graph("moving_rect", media_dir=fixtures_dir, kg_dir=tmp_dir, detector=detector, audio_tagger=tagger)
        g2 = build_graph("static_silent", media_dir=fixtures_dir, kg_dir=tmp_dir, detector=detector, audio_tagger=tagger)

        # Node IDs must be unique to each video
        nodes1 = set(g1["nodes"].keys())
        nodes2 = set(g2["nodes"].keys())
        assert len(nodes1 & nodes2) == 0, f"Node IDs overlapped between videos: {nodes1 & nodes2}"

        # Distinct acoustic envelopes
        aud1 = [n["loudness"] for n in g1["nodes"].values() if n["type"] == "audio_event"]
        aud2 = [n["loudness"] for n in g2["nodes"].values() if n["type"] == "audio_event"]
        assert aud1 != aud2, "Audio loudness profiles were unexpectedly identical across different videos!"


def test_tools_raise_without_mock_when_weights_missing():
    """FIX 5(a, b): Tools must raise FileNotFoundError/RuntimeError when weights are missing and use_mock=False."""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        ReferDINOTool(checkpoint_path="./nonexistent_gdino_ckpt.pth", use_mock=False)

    with pytest.raises((FileNotFoundError, RuntimeError)):
        SAM2Tool(sam2_checkpoint="./nonexistent_sam2_ckpt.pt", use_mock=False)


def test_kg_audioset_labels_and_classes():
    """Verify that KG audio nodes carry real AudioSet class names."""
    fixtures_dir = os.path.join(REPO_ROOT, "tests", "fixtures")
    from tests.test_kg import StubDetector, StubTagger
    detector = StubDetector()
    tagger = StubTagger(primary_label="Speech")

    with tempfile.TemporaryDirectory() as tmp_dir:
        g = build_graph("moving_rect", media_dir=fixtures_dir, kg_dir=tmp_dir, detector=detector, audio_tagger=tagger)
        audio_nodes = [n for n in g["nodes"].values() if n["type"] == "audio_event"]
        assert len(audio_nodes) > 0
        for an in audio_nodes:
            assert "label" in an
            assert isinstance(an["label"], str)
            assert len(an["label"]) > 0


def test_use_kg_false_rejection_in_trajectory():
    """Verify that when use_kg=False, retrieve_evidence action is marked invalid."""
    from rl_grpo.trajectory import TrajectoryCollector
    from rl_grpo.model_adapter import TGSModelAdapter, ModelAction

    class FakeAdapter:
        def sample_action(self, *args, **kwargs):
            return ModelAction(
                action_dict={"thought": "find kg", "action": "retrieve_evidence", "args": {"query": "dog"}},
                action_name="retrieve_evidence",
                args={"query": "dog"},
                thought="find kg",
                raw_text='{"thought": "find kg", "action": "retrieve_evidence", "args": {"query": "dog"}}',
                is_valid=True,
                error=None,
                prompt_ids=[1],
                completion_ids=[2],
                old_logps=[-0.1]
            )

    collector = TrajectoryCollector(model_adapter=FakeAdapter(), use_kg=False, max_turns=1)
    sample = {"uid": "test_sample_01", "expression": "dog", "video": None, "audio": None}
    traj = collector.rollout(sample)

    assert len(traj.turns) == 1
    t0 = traj.turns[0]
    assert t0.action == "retrieve_evidence"
    assert t0.is_valid is False
    assert "disabled when --use_kg false" in t0.observation


def test_grounding_tool_switch():
    """Verify grounder switch between gdino and referdino."""
    from tools.referdino_tool import get_grounding_tool

    gdino_tool = get_grounding_tool("gdino", use_mock=True)
    referdino_tool = get_grounding_tool("referdino", use_mock=True)

    cands_g = gdino_tool.run("test_vid", "a dog")
    cands_r = referdino_tool.run("test_vid", "a dog")

    assert len(cands_g) > 0
    assert len(cands_r) > 0
    assert hasattr(cands_g[0], "boxes")
    assert hasattr(cands_r[0], "boxes")


def test_referdino_raises_without_checkpoint():
    """ReferDINOTool must raise FileNotFoundError when checkpoint is missing and use_mock=False."""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        ReferDINOTool(checkpoint_path="nonexistent_referdino_ckpt.pth", use_mock=False)


def test_referdino_candidate_schema_with_stub():
    """ReferDINOTool returns valid Candidate objects with an injected stub model."""
    fixtures_dir = os.path.join(REPO_ROOT, "tests", "fixtures")

    class ReferDINOStubModel:
        def predict_clip(self, frame_paths, text):
            return [
                {
                    "boxes": [[0.1, 0.2, 0.5, 0.6]] * 10,
                    "masks": None,
                    "score": 0.88,
                },
                {
                    "boxes": [[0.6, 0.3, 0.9, 0.8]] * 10,
                    "masks": None,
                    "score": 0.72,
                }
            ]

    tool = ReferDINOTool(
        checkpoint_path="dummy",
        media_dir=fixtures_dir,
        use_mock=False,
        model=ReferDINOStubModel(),
    )

    cands = tool.run("moving_rect", "the moving object")
    assert len(cands) == 2
    for c in cands:
        assert isinstance(c, Candidate)
        assert hasattr(c, "cand_id")
        assert len(c.boxes) == 10
        assert hasattr(c, "score")
        assert hasattr(c, "mean_area")
        assert c.side in ("left", "center", "right")


def test_referdino_tool_no_groundingdino_import():
    """Verify that ReferDINOTool does NOT import grounding_dino or groundingdino."""
    referdino_file = os.path.join(REPO_ROOT, "tools", "referdino_tool.py")
    with open(referdino_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ReferDINOTool")
    imports = [
        alias.name.lower()
        for node in ast.walk(cls)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imports.extend(
        node.module.lower()
        for node in ast.walk(cls)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any("groundingdino" in name or "grounding_dino" in name for name in imports)


def test_server_info_endpoint():
    """Verify RealToolServer /info endpoint returns grounder information."""
    from tools.server import RealToolServer
    server = RealToolServer(media_dir="./REFAVS/media", grounder="gdino", use_mock=True)
    assert hasattr(server, "grounder_name")
    assert server.grounder_name == "gdino"
    assert server.referdino is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
