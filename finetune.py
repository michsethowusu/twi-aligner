#!/usr/bin/env python3
"""Thin wrapper so the documented `python finetune.py` keeps working.

The implementation lives in twi_aligner/finetune.py. Run this from the
repository root (so data/, models/, and output/ resolve correctly).
"""
from twi_aligner.finetune import main

if __name__ == "__main__":
    main()
