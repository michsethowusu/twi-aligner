"""Shared project paths and constants.

All paths are relative to the current working directory, which is expected to
be the repository root — the same convention the CLI has always used
(`python align.py` from the repo root). This keeps the simple "drop files in
data/, run, read output/" workflow intact.
"""
from pathlib import Path

MODEL_DIR  = Path("models")
AUDIO_DIR  = Path("data/audio")
TEXT_DIR   = Path("data/text")
OUTPUT_DIR = Path("output")

# Audio files longer than this (seconds) are auto-segmented before alignment.
# Forced alignment is all-or-nothing per utterance and grows fragile on long
# clips, so the default is deliberately short. Override with --max-seconds.
MAX_UTTERANCE_SECONDS = 15
