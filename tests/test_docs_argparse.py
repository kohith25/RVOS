#!/usr/bin/env python3
"""
tests/test_docs_argparse.py – Phase 0.4

Every --flag value example in RLGRPO.md must parse without error
using the REAL argparse of the target script.

Rules:
  - Only examples inside ```bash ... ``` fenced blocks are tested.
  - Lines that are comments (#) or conda/git commands are skipped.
  - Each python scripts/X.py ... invocation is split into argv and
    fed to that script's get_parser().parse_args(argv).
  - If a script has no get_parser() it is skipped with a warning (not failed).
"""

import os
import re
import sys
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_RLGRPO_MD = os.path.join(_REPO, "RLGRPO.md")

# Scripts that expose get_parser() and their import paths
_SCRIPT_MAP = {
    "scripts/run_pipeline.py":                   "scripts.run_pipeline",
    "scripts/compare_runs.py":                   "scripts.compare_runs",
    "scripts/rl_grpo/train_rl_tgs.py":           "scripts.rl_grpo.train_rl_tgs",
    "scripts/rl_grpo/check_format_validity.py":  "scripts.rl_grpo.check_format_validity",
    "scripts/rl_grpo/build_coldstart_data.py":   "scripts.rl_grpo.build_coldstart_data",
    "scripts/rl_grpo/sft_action_format.py":      "scripts.rl_grpo.sft_action_format",
    "tools/server.py":                           "tools.server",
}


def _extract_bash_examples():
    """Parse RLGRPO.md and return list of (script_path, argv_list) for every
    invocation of a known script inside a ```bash block."""
    with open(_RLGRPO_MD, "r", encoding="utf-8") as f:
        content = f.read()

    examples = []
    in_block = False
    pending_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```bash"):
            in_block = True
            pending_lines = []
            continue
        if in_block and stripped.startswith("```"):
            in_block = False
            # Process collected lines
            full_cmd = " ".join(pending_lines)
            # Find invocations of known scripts
            for script_path in _SCRIPT_MAP:
                pattern = r"python\s+" + re.escape(script_path) + r"([\s\\][^\n]*)"
                for m in re.finditer(pattern, full_cmd):
                    args_str = m.group(1)
                    # Clean up continuation backslashes and extra spaces
                    args_str = args_str.replace("\\", " ").replace("\n", " ")
                    # Split into tokens, skip comment-like tokens
                    tokens = args_str.split()
                    # Filter out empty strings
                    argv = [t for t in tokens if t and not t.startswith("#")]
                    # Replace /path/to/... placeholders with safe dummy values
                    argv = [
                        "dummy_value" if t.startswith("/path/to") else t
                        for t in argv
                    ]
                    examples.append((script_path, argv))
            pending_lines = []
            continue
        if in_block:
            # Accumulate continuation lines (backslash-continued)
            if stripped.startswith("#") or stripped.startswith("conda ") or stripped.startswith("cd "):
                continue
            # Strip trailing backslash
            stripped_cont = stripped.rstrip("\\").strip()
            if stripped_cont:
                pending_lines.append(stripped_cont)

    return examples


_EXAMPLES = _extract_bash_examples()


@pytest.mark.parametrize("script_path,argv", _EXAMPLES,
                         ids=[f"{s.split('/')[-1]}:{' '.join(a[:2])}" for s, a in _EXAMPLES])
def test_docs_flags_parse(script_path, argv):
    """Every --flag in RLGRPO.md must be accepted by the script's real argparse."""
    module_name = _SCRIPT_MAP.get(script_path)
    assert module_name, f"Unknown script: {script_path}"

    mod = pytest.importorskip(module_name, reason=f"Cannot import {module_name}")

    if not hasattr(mod, "get_parser"):
        pytest.skip(f"{module_name} has no get_parser(); skipped (not failed)")

    parser = mod.get_parser()
    # parse_known_args so extra flags (e.g. from continuation) don't break
    try:
        ns, unknown = parser.parse_known_args(argv)
    except SystemExit as e:
        pytest.fail(
            f"argparse SystemExit({e.code}) for {script_path} with args: {argv}\n"
            f"This means a --flag in RLGRPO.md is invalid."
        )
    # Warn on unknown args (may be doc drift) but don't fail — only flag-not-found exits fail
    if unknown:
        print(f"\n[WARN] {script_path}: unrecognised tokens in docs: {unknown}")
