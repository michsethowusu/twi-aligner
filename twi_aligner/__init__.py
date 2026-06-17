"""Twi Forced Aligner — word-level forced alignment for Twi (Akan) audio.

Public submodules:
    align        – the alignment pipeline (run via `python align.py`)
    dataset      – batch alignment from a HF dataset or CSV
    finetune     – adapt the acoustic model to your own data
    g2p          – grapheme-to-phoneme conversion for out-of-vocabulary words
    download     – model/dictionary download from GitHub Releases
    audio        – audio conversion and auto-segmentation helpers
"""

__version__ = "1.0.0"
