#!/usr/bin/env python3
"""
tests/test_file_hashes.py - Byte-Identity Verification Test with CRLF Normalization

Fails if ANY original file in the TGS-Agent repository has been modified.
Uses a stored SHA-256 manifest of the original files computed with line endings normalized
(b'\\r\\n' -> b'\\n') to ensure reliable operation on Windows CRLF checkouts.
"""

import os
import hashlib
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Stored SHA-256 manifest computed with line-endings normalized
NORMALIZED_MANIFEST = {
    "scripts/finetune/inference_hyper_lora.py": "485b83c8eb360b98d067510ed0012cbc35b267bd9ebaa7703d1e4ad5446fc086",
    "scripts/finetune/inference_hyper_lora.sh": "e359bbcaa19ee35e300426dd3f1c93f175760cda4298bb35cd611a6d0f3dd1aa",
    "scripts/finetune/finetune_hyperlora.py": "9b1c81a5d6f393c987fd0d12087beba43072650f8b5285b6b7a6728210461241",
    "dataset/unified_dataset.py": "790952867752b4796ac81d205c615d0e824520d4fa0b1ef79ecca55e494b6679",
    "models/unified_arch.py": "2ee83ba9e233ddcf161b459a2718b1461d852274a113fb1ad7ec08c036acb1c7",
    "models/unified_llama.py": "aac85d18619f6aa8a5f0608d6c0b57b973e0c538951be1d252df7855d3f78a95",
    "utils/util.py": "732f8093654af84c9a8c1ac8a28418c89fa4fda17e95b922b7747904e6f48f65",
    "utils/avss_utils.py": "6e8f237da2b64c82551ced8c3af1e2b984ee41ee365f9e1bc315a79906410056",
    "ground_segment_scripts/ground_segment_with_object_text_after_thinking_for_RefAVSBench.py": "b8d31577078cba7297804fc1a8355ef54b70b8b8b444cc3650130b7df82e5943",
    "ground_segment_scripts/ground_segment_with_object_text_after_thinking_for_R2AVSBench.py": "361cd98ed14fd96e3518b9f8fa2e2a419da4a30a1a35241c77169603b9602107",
    "ground_segment_scripts/ground_segment_with_direct_reference_of_RefAVSBench.py": "bd6dfa8f1dee1e68c518b014238d1582f194adb341a669b96482b1d27e146f24",
    "ground_segment_scripts/ground_segment_with_direct_reference_of_R2AVSBench.py": "05d47f66b33cad469357ca1ba70e179415b54ae54a7289135153ec6e677ed70d",
    "peft_hyper/peft_model.py": "b814d1e346735660da20ff39c1e6437858203e0837795672f1fc5aa3a6ea360f",
    "README.md": "cb0796bb3297720a6346e1352ad5494be392b7277aaa7b2e904e64e55c3d02a9",
    "think_environment.yml": "9fb02c81feb8cc2e1760eb6c26367c9176228f257dfcbc0bff1975641a1d1897",
    "dinosam2_environment.yml": "7e1abd09dae7de64937c015722bf12f6c03ff72289b30d7078ed9f0ba8b1ccd7",
}


def compute_normalized_sha256(file_path):
    with open(file_path, "rb") as f:
        content = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest().lower()


@pytest.mark.parametrize("rel_path, expected_hash", NORMALIZED_MANIFEST.items())
def test_original_file_hashes(rel_path, expected_hash):
    """Verify each original file has exact byte-identical hash (line endings normalized)."""
    full_path = os.path.join(REPO_ROOT, rel_path)
    assert os.path.exists(full_path), f"Original file {rel_path} does not exist at {full_path}"
    actual_hash = compute_normalized_sha256(full_path)
    assert actual_hash == expected_hash, (
        f"File {rel_path} hash mismatch!\n"
        f"Expected: {expected_hash}\n"
        f"Actual:   {actual_hash}\n"
        "Original TGS-Agent files must stay strictly byte-identical!"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
