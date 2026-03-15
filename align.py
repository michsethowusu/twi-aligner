#!/usr/bin/env python3
"""
align_dataset.py
================
Align a dataset word-by-word using the Twi Forced Aligner.  Two input modes
are supported:

  HF mode   – download an audio-text dataset directly from Hugging Face
  CSV mode  – use a local CSV/TSV that maps audio file paths to transcripts

In both cases the script prepares the data, runs align.py, parses the output
TextGrid files, and writes a TSV with one row per aligned word.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HF MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python align_dataset.py \\
      --dataset Ghana/twi-religious-speech \\
      --split train \\
      --audio-col audio \\
      --text-col transcription \\
      --max-samples 100 \\
      --output-tsv alignments.tsv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSV MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The CSV/TSV must have at least two columns: one for audio paths and one for
transcripts.  Paths can be absolute or relative to the CSV file's directory.

  python align_dataset.py \\
      --csv metadata.csv \\
      --audio-col path \\
      --text-col sentence \\
      --output-tsv alignments.tsv

Example metadata.csv:
  path,sentence
  recordings/001.wav,meda wo ase
  recordings/002.mp3,ɛte sɛn

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT TSV COLUMNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  sample_id   word   start_sec   end_sec   duration_sec
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import textgrid
    TEXTGRID_AVAILABLE = True
except ImportError:
    TEXTGRID_AVAILABLE = False

# ── Paths (must match align.py) ────────────────────────────────────────────────
MODEL_DIR  = Path("models")
AUDIO_DIR  = Path("data/audio")
TEXT_DIR   = Path("data/text")
OUTPUT_DIR = Path("output")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def check_models() -> bool:
    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt  = MODEL_DIR / "twi_lexicon.txt"
    if not model_zip.exists() or not dict_txt.exists():
        print("❌ Model or lexicon not found in models/.")
        print("   Run:  python align.py --update   to download them first.")
        return False
    return True


def sanitise_id(raw: str) -> str:
    """Turn any string into a safe filename stem (alphanumeric + underscores)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(raw))


def prepare_data_dirs() -> None:
    for d in (AUDIO_DIR, TEXT_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    # Clear leftover files from any previous run so MFA starts clean.
    for p in list(AUDIO_DIR.iterdir()) + list(TEXT_DIR.iterdir()):
        if p.is_file():
            p.unlink()


def write_txt(sample_id: str, text: str) -> None:
    (TEXT_DIR / f"{sample_id}.txt").write_text(text.strip(), encoding="utf-8")


def unique_id(base: str, seen: set) -> str:
    """Append a counter suffix if base is already taken."""
    sid = base
    counter = 1
    while sid in seen:
        sid = f"{base}_{counter:04d}"
        counter += 1
    seen.add(sid)
    return sid


# ── HF mode ────────────────────────────────────────────────────────────────────

def load_hf(args) -> int:
    """Download HF dataset, write wav+txt pairs.  Returns number written."""
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("❌ 'datasets' not installed.  Run:  pip install datasets")
        sys.exit(1)

    if not SOUNDFILE_AVAILABLE:
        print("❌ 'soundfile' not installed.  Run:  pip install soundfile")
        sys.exit(1)

    print(f"\n📥 Loading '{args.dataset}' (split: {args.split})...")
    try:
        ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        sys.exit(1)

    try:
        ds = ds.cast_column(args.audio_col, Audio(sampling_rate=16_000))
    except Exception as e:
        print(f"❌ Could not cast audio column '{args.audio_col}': {e}")
        sys.exit(1)

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print(f"  ✓ {len(ds)} sample(s) to process.")
    prepare_data_dirs()

    written, seen = 0, set()
    for idx, sample in enumerate(ds):
        text = sample.get(args.text_col, "").strip()
        if not text:
            print(f"  ⚠ Sample {idx}: empty text – skipping.")
            continue

        audio_data = sample.get(args.audio_col)
        if audio_data is None:
            print(f"  ⚠ Sample {idx}: no audio – skipping.")
            continue

        raw_id = sample.get("id", sample.get("path", f"sample_{idx:05d}"))
        sid    = unique_id(sanitise_id(Path(str(raw_id)).stem), seen)

        try:
            sf.write(
                str(AUDIO_DIR / f"{sid}.wav"),
                audio_data["array"],
                audio_data["sampling_rate"],
                subtype="PCM_16",
            )
        except Exception as e:
            print(f"  ⚠ Could not write audio for {sid}: {e}")
            continue

        write_txt(sid, text)
        written += 1

    return written


# ── CSV mode ───────────────────────────────────────────────────────────────────

def load_csv(args) -> int:
    """
    Read a local CSV/TSV, copy audio files into data/audio/ (converting via
    ffmpeg if needed), and write transcripts into data/text/.
    Returns number of samples written.
    """
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)

    delimiter = "\t" if csv_path.suffix.lower() in (".tsv", ".tab") else ","

    print(f"\n📄 Reading '{csv_path}'...")
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if args.audio_col not in (reader.fieldnames or []):
            print(f"❌ Audio column '{args.audio_col}' not found in CSV.")
            print(f"   Available columns: {reader.fieldnames}")
            sys.exit(1)
        if args.text_col not in (reader.fieldnames or []):
            print(f"❌ Text column '{args.text_col}' not found in CSV.")
            print(f"   Available columns: {reader.fieldnames}")
            sys.exit(1)
        rows = list(reader)

    if args.max_samples:
        rows = rows[: args.max_samples]

    print(f"  ✓ {len(rows)} row(s) to process.")
    prepare_data_dirs()

    # Resolve audio paths relative to the CSV's directory
    csv_dir = csv_path.parent

    written, seen = 0, set()
    for idx, row in enumerate(rows):
        text = row.get(args.text_col, "").strip()
        if not text:
            print(f"  ⚠ Row {idx}: empty text – skipping.")
            continue

        raw_audio_path = row.get(args.audio_col, "").strip()
        if not raw_audio_path:
            print(f"  ⚠ Row {idx}: empty audio path – skipping.")
            continue

        src = Path(raw_audio_path)
        if not src.is_absolute():
            src = csv_dir / src
        if not src.exists():
            print(f"  ⚠ Audio file not found: {src} – skipping.")
            continue

        sid  = unique_id(sanitise_id(src.stem), seen)
        dest = AUDIO_DIR / f"{sid}.wav"

        # Copy as-is if already a WAV; otherwise let align.py's ffmpeg step
        # handle conversion (it re-encodes everything to 16 kHz mono anyway).
        try:
            if src.suffix.lower() == ".wav":
                shutil.copy2(src, dest)
            else:
                # Pre-convert non-WAV formats so MFA can find a .wav file
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(src),
                     "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dest)],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"  ⚠ ffmpeg could not convert {src.name}: "
                          f"{result.stderr.strip()[-150:]}")
                    continue
        except Exception as e:
            print(f"  ⚠ Could not prepare audio for row {idx}: {e}")
            continue

        write_txt(sid, text)
        written += 1

    return written


# ── Alignment ──────────────────────────────────────────────────────────────────

def run_align(overwrite: bool) -> bool:
    cmd = [sys.executable, "align.py"]
    if overwrite:
        cmd.append("--overwrite")
    print("\n🚀 Running align.py...")
    return subprocess.run(cmd).returncode == 0


# ── TextGrid parsing ───────────────────────────────────────────────────────────

def parse_textgrid_lib(tg_path: Path) -> list[dict]:
    tg = textgrid.TextGrid.fromFile(str(tg_path))
    words = []
    for tier in tg.tiers:
        if tier.name.lower() in ("words", "word"):
            for interval in tier:
                label = interval.mark.strip()
                if label and label not in ("<eps>", "sp", "sil", ""):
                    words.append({
                        "word":         label,
                        "start_sec":    round(interval.minTime, 4),
                        "end_sec":      round(interval.maxTime, 4),
                        "duration_sec": round(interval.maxTime - interval.minTime, 4),
                    })
    return words


def parse_textgrid_manual(tg_path: Path) -> list[dict]:
    """Dependency-free TextGrid parser for MFA's standard output format."""
    text = tg_path.read_text(encoding="utf-8", errors="replace")
    words = []
    tier_blocks = re.split(r'item\s*\[\d+\]', text)
    word_tier = next(
        (b for b in tier_blocks if re.search(r'name\s*=\s*"words?"', b, re.IGNORECASE)),
        None,
    )
    if not word_tier:
        return words
    for xmin, xmax, label in re.findall(
        r'xmin\s*=\s*([\d.]+).*?xmax\s*=\s*([\d.]+).*?(?:text|mark)\s*=\s*"([^"]*)"',
        word_tier, re.DOTALL,
    ):
        label = label.strip()
        if label and label not in ("<eps>", "sp", "sil", ""):
            start = round(float(xmin), 4)
            end   = round(float(xmax), 4)
            words.append({
                "word":         label,
                "start_sec":    start,
                "end_sec":      end,
                "duration_sec": round(end - start, 4),
            })
    return words


def parse_textgrid(tg_path: Path) -> list[dict]:
    if TEXTGRID_AVAILABLE:
        try:
            return parse_textgrid_lib(tg_path)
        except Exception as e:
            print(f"  ⚠ textgrid library failed on {tg_path.name} ({e}), using fallback.")
    return parse_textgrid_manual(tg_path)


# ── Output ─────────────────────────────────────────────────────────────────────

def write_tsv(output_tsv: Path) -> int:
    tg_files = sorted(OUTPUT_DIR.glob("**/*.TextGrid"))
    if not tg_files:
        print("❌ No TextGrid files found in output/. Alignment produced no results.")
        return 0

    print(f"\n📄 Parsing {len(tg_files)} TextGrid file(s)...")
    total = 0
    with open(output_tsv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "word", "start_sec", "end_sec", "duration_sec"],
            delimiter="\t",
        )
        writer.writeheader()
        for tg_path in tg_files:
            words = parse_textgrid(tg_path)
            if not words:
                print(f"  ⚠ No word intervals in {tg_path.name}")
                continue
            for w in words:
                writer.writerow({"sample_id": tg_path.stem, **w})
            total += len(words)

    return total


def print_preview(output_tsv: Path, n: int = 15) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {'SAMPLE_ID':<25} {'WORD':<16} {'START':>8}  {'END':>8}")
    print(f"{'─' * 62}")
    with open(output_tsv, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter="\t")):
            if i >= n:
                break
            print(
                f"  {row['sample_id']:<25} {row['word']:<16} "
                f"{float(row['start_sec']):>8.3f}  {float(row['end_sec']):>8.3f}"
            )
    print(f"{'─' * 62}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Align a dataset word-by-word: HF dataset or local CSV + audio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input – mutually exclusive modes
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", metavar="HF_REPO_ID",
                     help="Hugging Face dataset repo ID  (HF mode)")
    src.add_argument("--csv",     metavar="FILE",
                     help="Local CSV/TSV metadata file   (CSV mode)")

    # Shared column names
    parser.add_argument("--audio-col",   default="audio",
                        help="Column with audio data/paths (default: audio)")
    parser.add_argument("--text-col",    default="sentence",
                        help="Column with transcripts     (default: sentence)")

    # HF-only
    parser.add_argument("--split",       default="train",
                        help="[HF] Dataset split           (default: train)")

    # Common
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples processed (useful for testing)")
    parser.add_argument("--output-tsv",  default="alignments.tsv",
                        help="Output TSV path              (default: alignments.tsv)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite existing alignment files")
    parser.add_argument("--keep-data",   action="store_true",
                        help="Keep prepared files in data/audio/ and data/text/")

    args = parser.parse_args()

    if not check_models():
        sys.exit(1)

    if not TEXTGRID_AVAILABLE:
        print("ℹ  'textgrid' library not found — using built-in parser.")
        print("   For more robust parsing:  pip install TextGrid\n")

    # ── Load data ───────────────────────────────────────────────────────────────
    if args.dataset:
        written = load_hf(args)
    else:
        written = load_csv(args)

    if written == 0:
        print("❌ No samples were prepared. Nothing to align.")
        sys.exit(1)
    print(f"\n  ✓ {written} sample(s) ready for alignment.")

    # ── Align ───────────────────────────────────────────────────────────────────
    if not run_align(overwrite=args.overwrite):
        print("\n❌ Alignment failed. See MFA output above.")
        sys.exit(1)

    # ── Parse & write TSV ───────────────────────────────────────────────────────
    output_tsv = Path(args.output_tsv)
    total_words = write_tsv(output_tsv)
    if total_words == 0:
        sys.exit(1)

    print(f"  ✓ {total_words} word alignment(s) written to {output_tsv}")
    print_preview(output_tsv)
    print(f"\n✅ Done.  Full results in: {output_tsv}")

    # ── Cleanup ─────────────────────────────────────────────────────────────────
    if not args.keep_data:
        for p in list(AUDIO_DIR.iterdir()) + list(TEXT_DIR.iterdir()):
            if p.is_file():
                p.unlink()
        print("  (Temp files cleaned up.  Use --keep-data to retain them.)")


if __name__ == "__main__":
    main()
