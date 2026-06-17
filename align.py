#!/usr/bin/env python3
"""Thin wrapper so the documented `python align.py` keeps working.

The implementation lives in twi_aligner/align.py. Run this from the
repository root (so data/, models/, and output/ resolve correctly).
"""
from twi_aligner.align import main

if __name__ == "__main__":
    main()
