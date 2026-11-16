#!/usr/bin/env python3
"""Verify the original-file hash manifest against its canonical commit."""

import ast
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMMIT = "2f711e53ee05dfb5c6c51852ca4155d8e292209c"
MANIFEST_PATH = "tests/test_file_hashes.py"
MANIFEST_NAME = "NORMALIZED_MANIFEST"
UPDATE_MARKER = "UPDATE_HASH_MANIFEST"


def load_manifest(source: str) -> dict:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == MANIFEST_NAME
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError(f"{MANIFEST_NAME} must be a dictionary")
            return value
    raise ValueError(f"{MANIFEST_NAME} was not found in {MANIFEST_PATH}")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
        encoding="utf-8",
    )


def main() -> int:
    canonical_source = git_output(
        "show", f"{CANONICAL_COMMIT}:{MANIFEST_PATH}"
    )
    current_source = (REPO_ROOT / MANIFEST_PATH).read_text(encoding="utf-8")
    canonical_manifest = load_manifest(canonical_source)
    current_manifest = load_manifest(current_source)

    if current_manifest == canonical_manifest:
        print("Original-file hash manifest matches the canonical commit.")
        return 0

    commit_message = git_output("log", "-1", "--format=%B")
    if UPDATE_MARKER in commit_message:
        print(f"Manifest update authorized by {UPDATE_MARKER}.")
        return 0

    missing = sorted(set(canonical_manifest) - set(current_manifest))
    added = sorted(set(current_manifest) - set(canonical_manifest))
    changed = sorted(
        path for path in set(canonical_manifest) & set(current_manifest)
        if canonical_manifest[path] != current_manifest[path]
    )
    print(
        "Original-file hash manifest differs from the canonical commit. "
        f"Missing: {missing}; added: {added}; changed: {changed}. "
        f"Use {UPDATE_MARKER} in the commit message only when intentionally "
        "updating the original-file baseline.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
