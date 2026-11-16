import csv
import os
import tempfile

from kg.build_kg import (
    DEFAULT_OBJECT_VOCAB,
    build_detector_prompts,
    build_graph,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURES_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")


class RecordingDetector:
    def __init__(self):
        self.prompts = []

    def detect_frame(self, frame_path, text_prompt):
        self.prompts.append(text_prompt)
        return []


class FixtureTagger:
    def extract_events(self, audio_path):
        return [
            {
                "second": second,
                "interval": [second, second + 1],
                "labels": ["Speech"],
                "primary_label": "Speech",
            }
            for second in range(10)
        ]


def _write_metadata(path, category):
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["vid", "uid", "split", "fid", "exp"])
        writer.writerow(["moving_rect", f"moving_rect_{category}_0", "train", "0", ""])


def test_all_51_classes_are_batched_with_reference_terms():
    detector = RecordingDetector()
    prompts = build_detector_prompts(["guitar", "left"], DEFAULT_OBJECT_VOCAB)
    assert len(DEFAULT_OBJECT_VOCAB) == 51
    assert len(prompts) > 1
    for label in DEFAULT_OBJECT_VOCAB:
        assert any(label in prompt for prompt in prompts), label
    for prompt in prompts:
        assert "guitar" in prompt
        assert "left" in prompt
        assert len(prompt.split()) <= 256

    frame_paths = [os.path.join(FIXTURES_DIR, "moving_rect", "frames", f"{i}.jpg") for i in range(10)]
    from kg.build_kg import run_grounding_dino_on_frames
    run_grounding_dino_on_frames(frame_paths, prompts, detector=detector)
    assert set(detector.prompts) == set(prompts)


def test_cache_rebuilds_when_vocabulary_hash_changes(tmp_path):
    metadata_a = tmp_path / "metadata_a.csv"
    metadata_b = tmp_path / "metadata_b.csv"
    _write_metadata(metadata_a, "guitar")
    _write_metadata(metadata_b, "violin")
    detector = RecordingDetector()
    kwargs = {
        "video_id": "moving_rect",
        "media_dir": FIXTURES_DIR,
        "kg_dir": str(tmp_path / "kg_cache"),
        "detector": detector,
        "audio_tagger": FixtureTagger(),
    }

    first = build_graph(**kwargs, meta_csv_path=str(metadata_a))
    calls_after_first = len(detector.prompts)
    second = build_graph(**kwargs, meta_csv_path=str(metadata_b))

    assert first["config_hash"] != second["config_hash"]
    assert len(detector.prompts) > calls_after_first
