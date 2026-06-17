#!/usr/bin/env python3
"""Thin wrapper so the documented `python align_dataset.py` keeps working.

The implementation lives in twi_aligner/dataset.py. Run this from the
repository root (so data/, models/, and output/ resolve correctly).
"""
from twi_aligner.dataset import main

if __name__ == "__main__":
    main()
