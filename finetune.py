#!/usr/bin/env python3
"""
Twi Forced Aligner – Finetuning Script
=======================================
Adapts the pre-trained GMM-HMM acoustic model to your own Twi data using
MFA's built-in `adapt` command (a.k.a. MAP adaptation / fMLLR).

This is the recommended workflow when the base model — which was trained on
religious speech — performs poorly on your domain (e.g. conversational Twi,
broadcast, storytelling, etc.).

How it works
------------
MFA's acoustic models are Kaldi-based GMM-HMM models.  "Finetuning" them
means running Maximum A Posteriori (MAP) adaptation and feature-space
Maximum Likelihood Linear Regression (fMLLR) on your labelled data.
This updates the Gaussian mixture parameters to better fit your speaker(s)
and domain without discarding what the base model already learned.

Requirements
------------
- At least ~15–30 minutes of aligned, transcribed Twi audio for adaptation.
  More data = better results; 1–2 hours is ideal.
- The same conda environment used for align.py (MFA + ffmpeg).
- The base model downloaded by align.py (models/twi_acoustic_model.zip).

Usage
-----
  python finetune.py                          # adapt with data/finetune/
  python finetune.py --data-dir my_data/      # custom data directory
  python finetune.py --output-model my_model  # custom output model name
  python finetune.py --num-jobs 4             # parallelise across 4 CPUs
  python finetune.py --overwrite              # overwrite existing adapted model

Data format (same as align.py)
-------------------------------
  data/finetune/audio/  – .wav / .mp3 / .flac / .m4a / .ogg files
  data/finetune/text/   – matching .txt transcripts (UTF-8, one per audio file)

Output
------
  models/twi_acoustic_model_adapted.zip   (or the name set by --output-model)

After adaptation, point align.py at the new model:
  Place the adapted zip in models/ and rename it twi_acoustic_model.zip,
  or edit the MODEL_DIR / model filename at the top of align.py.
"""

import re
import sys
import shutil
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_DIR         = Path("models")
DEFAULT_DATA_DIR  = Path("data/finetune")
DEFAULT_OUT_MODEL = "twi_acoustic_model_adapted"
# ------------------------------------------------------------------------------


def check_mfa() -> bool:
    try:
        subprocess.run(["mfa", "version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def get_audio_duration(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def convert_audio(audio_dir: Path) -> None:
    """Re-encode all audio to 16 kHz mono 16-bit PCM WAV, in-place."""
    if not check_ffmpeg():
        print("⚠ ffmpeg not found – skipping audio conversion.")
        print("  Install with:  conda install -c conda-forge ffmpeg")
        return

    supported = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
    files = [p for p in audio_dir.iterdir() if p.suffix.lower() in supported]
    if not files:
        return

    print(f"\n🔄 Converting {len(files)} audio file(s) to 16 kHz mono WAV...")
    converted = 0
    for src in files:
        dest = src.with_suffix(".wav")
        tmp  = dest.with_suffix(".tmp.wav")
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src),
                 "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(tmp)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"  ⚠ Could not convert {src.name}: {r.stderr.strip()[-200:]}")
                tmp.unlink(missing_ok=True)
                continue
            if src != dest:
                src.unlink()
            tmp.replace(dest)
            converted += 1
        except Exception as e:
            print(f"  ⚠ Error converting {src.name}: {e}")
            tmp.unlink(missing_ok=True)
    print(f"  ✓ {converted} file(s) converted/verified.")


def total_audio_duration(audio_dir: Path) -> float:
    return sum(
        get_audio_duration(p) or 0.0
        for p in audio_dir.glob("*.wav")
    )


def validate_pairs(audio_dir: Path, text_dir: Path) -> bool:
    audio_stems = {p.stem for p in audio_dir.glob("*.wav")}
    text_stems  = {p.stem for p in text_dir.glob("*.txt")}
    matched     = audio_stems & text_stems

    for stem in sorted(audio_stems - text_stems):
        print(f"  ⚠ {stem}.wav has no transcript – will be skipped.")
    for stem in sorted(text_stems - audio_stems):
        print(f"  ⚠ {stem}.txt has no audio – will be skipped.")

    if not matched:
        print("❌ No matched audio/text pairs found in the finetune data.")
        return False

    print(f"  ✓ {len(matched)} matched pair(s) found.")
    return True


def copy_texts_alongside_audio(text_dir: Path, audio_dir: Path) -> None:
    """MFA expects transcripts in the same directory as audio."""
    for txt in text_dir.glob("*.txt"):
        dest = audio_dir / txt.name
        if not dest.exists():
            shutil.copy2(str(txt), dest)


def run_finetune(
    data_dir: Path,
    output_model_name: str,
    num_jobs: int,
    overwrite: bool,
) -> None:
    audio_dir = data_dir / "audio"
    text_dir  = data_dir / "text"
    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt  = MODEL_DIR / "twi_lexicon.txt"
    out_zip   = MODEL_DIR / f"{output_model_name}.zip"

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    errors = False

    if not check_mfa():
        print("❌ MFA not found. Install with:")
        print("   conda install -c conda-forge montreal-forced-aligner")
        errors = True

    if not model_zip.exists():
        print("❌ Base model not found at models/twi_acoustic_model.zip")
        print("   Run:  python align.py --update   to download it first.")
        errors = True

    if not dict_txt.exists():
        print("❌ Lexicon not found at models/twi_lexicon.txt")
        print("   Run:  python align.py --update   to download it first.")
        errors = True

    if not audio_dir.exists() or not text_dir.exists():
        print(f"❌ Data directory structure not found.")
        print(f"   Expected:")
        print(f"     {audio_dir}/  – audio files")
        print(f"     {text_dir}/   – transcript .txt files")
        errors = True

    if errors:
        sys.exit(1)

    if out_zip.exists() and not overwrite:
        print(f"⚠ Adapted model already exists at {out_zip}")
        print("  Use --overwrite to replace it.")
        sys.exit(0)

    # ── Prepare audio ──────────────────────────────────────────────────────────
    convert_audio(audio_dir)

    # ── Validate data ──────────────────────────────────────────────────────────
    print("\n🔍 Validating finetune data...")
    if not validate_pairs(audio_dir, text_dir):
        sys.exit(1)

    duration_secs = total_audio_duration(audio_dir)
    duration_mins = duration_secs / 60
    print(f"  Total audio: {duration_mins:.1f} minutes")

    if duration_mins < 15:
        print("\n⚠  Warning: less than 15 minutes of audio detected.")
        print("   Adaptation on very small datasets may degrade alignment quality.")
        print("   Consider collecting more data before finetuning.")
        response = input("   Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(0)

    # ── Co-locate transcripts with audio (MFA requirement) ────────────────────
    copy_texts_alongside_audio(text_dir, audio_dir)

    # ── Run MFA adapt ─────────────────────────────────────────────────────────
    adapt_output_dir = MODEL_DIR / "adapt_output"
    adapt_output_dir.mkdir(exist_ok=True)

    cmd = [
        "mfa", "adapt",
        str(audio_dir),
        str(dict_txt),
        str(model_zip),
        str(adapt_output_dir),
        "--output_model_path", str(out_zip),
        "--num_jobs", str(num_jobs),
        "--clean",
    ]
    if overwrite:
        cmd.append("--overwrite")

    print(f"\n🎯 Starting MAP adaptation with {num_jobs} job(s)...")
    print(f"   Base model : {model_zip}")
    print(f"   Lexicon    : {dict_txt}")
    print(f"   Data       : {audio_dir}")
    print(f"   Output     : {out_zip}\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Adaptation failed (exit code {e.returncode}).")
        print("\nTroubleshooting tips:")
        print("  1. Check the MFA log output above for details.")
        print("  2. Validate your data with:")
        print(f"     mfa validate {audio_dir} {dict_txt} {model_zip}")
        print("  3. Make sure all words in your transcripts are in the lexicon.")
        print("     Add missing words manually to models/twi_lexicon.txt:")
        print("       word p h o n e m e s")
        sys.exit(1)

    # ── Verify output ──────────────────────────────────────────────────────────
    if not out_zip.exists():
        print(f"\n⚠ MFA finished but {out_zip} was not created.")
        print("  Check the adapt_output/ directory for intermediate files.")
        sys.exit(1)

    print(f"\n✅ Adaptation complete!")
    print(f"   Adapted model saved to: {out_zip}")
    print(f"\nTo use the adapted model for alignment:")
    print(f"  Option A – replace the base model:")
    print(f"    cp {out_zip} models/twi_acoustic_model.zip")
    print(f"    python align.py")
    print(f"  Option B – run MFA directly:")
    print(f"    mfa align data/audio/ {dict_txt} {out_zip} output/")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Finetune the Twi acoustic model on your own data via MFA MAP adaptation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune.py
  python finetune.py --data-dir my_recordings/
  python finetune.py --output-model twi_broadcast_model --num-jobs 4
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Root directory containing audio/ and text/ subdirs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-model",
        default=DEFAULT_OUT_MODEL,
        help=f"Name for the output model zip in models/ (default: {DEFAULT_OUT_MODEL})",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=2,
        help="Number of parallel MFA jobs (default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing adapted model",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Twi Forced Aligner – Model Adaptation")
    print("=" * 60)

    run_finetune(
        data_dir=args.data_dir,
        output_model_name=args.output_model,
        num_jobs=args.num_jobs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
