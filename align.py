#!/usr/bin/env python3
"""
Twi Forced Aligner – downloads a pre-trained acoustic model from GitHub Releases
(if not already present) and aligns audio/text using MFA.

Long recordings are automatically segmented into short utterance-level clips
using proportional word-rate splitting before alignment, so you can hand in
any length of audio.
"""

import re
import sys
import shutil
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import requests
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration – change if you fork the repository
REPO = "GhanaNLP/twi-aligner"
MODEL_DIR  = Path("models")
AUDIO_DIR  = Path("data/audio")
TEXT_DIR   = Path("data/text")
OUTPUT_DIR = Path("output")

# Audio files longer than this (seconds) are auto-segmented before alignment.
MAX_UTTERANCE_SECONDS = 30
# ----------------------------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── GitHub model download ──────────────────────────────────────────────────────

def get_all_releases(repo: str) -> List[Dict]:
    releases, page = [], 1
    while True:
        url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            if not data:
                break
            releases.extend(data)
            page += 1
        except Exception as e:
            print(f"Error fetching releases: {e}")
            break
    return releases

def download_file(url: str, dest: Path, desc: str = None) -> None:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=desc or dest.name, total=total,
        unit="B", unit_scale=True, unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def select_release_interactive(releases: List[Dict]) -> Optional[Dict]:
    if not releases:
        return None
    if len(releases) == 1:
        return releases[0]
    print("\nMultiple model releases found. Please choose one:")
    for i, rel in enumerate(releases, 1):
        name = rel.get("name") or rel.get("tag_name")
        published = rel.get("published_at", "")[:10]
        print(f"  {i}. {name} ({published})")
    while True:
        try:
            choice = int(input("Enter number (or 0 to cancel): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(releases):
                return releases[choice - 1]
        except ValueError:
            pass
        print("Invalid choice, try again.")

def ensure_model_and_dict(repo: str, force_update: bool = False) -> bool:
    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt  = MODEL_DIR / "twi_lexicon.txt"

    if model_zip.exists() and dict_txt.exists() and not force_update:
        print("✓ Model and dictionary already present (use --update to re-download).")
        return True

    print("Fetching available model releases from GitHub...")
    releases = get_all_releases(repo)
    if not releases:
        print("❌ No releases found. Check the repository name or your network connection.")
        return False

    selected = select_release_interactive(releases)
    if not selected:
        print("Download cancelled.")
        return False

    tag = selected["tag_name"]
    print(f"Selected release: {tag}")
    assets = {a["name"]: a["browser_download_url"] for a in selected["assets"]}

    for name in ["twi_acoustic_model.zip", "twi_lexicon.txt"]:
        if name not in assets:
            print(f"❌ Required asset '{name}' not found in release {tag}.")
            return False

    download_file(assets["twi_acoustic_model.zip"], model_zip, desc="Model ZIP")
    download_file(assets["twi_lexicon.txt"],         dict_txt,  desc="Dictionary")
    print("✓ Model and dictionary downloaded successfully.")
    return True

# ── Audio utilities ────────────────────────────────────────────────────────────

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

def convert_audio_to_mfa_format(audio_dir: Path) -> None:
    """Re-encode all audio in audio_dir to 16 kHz mono 16-bit PCM WAV."""
    if not check_ffmpeg():
        print("⚠ ffmpeg not found – skipping audio conversion.")
        print("  Install it with:  conda install -c conda-forge ffmpeg")
        return

    supported = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
    files = [p for p in audio_dir.iterdir() if p.suffix.lower() in supported]
    if not files:
        return

    print(f"\n🔄 Checking/converting {len(files)} audio file(s) to 16 kHz mono WAV...")
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

# ── Auto-segmentation ──────────────────────────────────────────────────────────

def split_transcript_into_sentences(text: str) -> List[str]:
    """
    Turn a raw transcript into a list of sentences.
    If already multi-line, use those lines directly.
    Otherwise split on sentence-ending punctuation (.  !  ?).
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1:
        return lines
    raw = lines[0] if lines else text.strip()
    chunks = [c.strip() for c in re.split(r'(?<=[.!?])\s+', raw) if c.strip()]
    return chunks if chunks else [raw]

def build_proportional_segments(sentences: List[str],
                                 total_duration: float,
                                 max_seconds: float = MAX_UTTERANCE_SECONDS) -> List[Dict]:
    """
    Split a transcript + audio into MFA-ready clips using word rate.

    Core idea:
      - Compute words-per-second: total_words / total_duration
      - Each word therefore takes total_duration / total_words seconds
      - Accumulate sentences until the running total would exceed max_seconds,
        then flush to a clip. Sentences are only used as 'do not cut mid-sentence'
        boundaries — they do not affect the overall word rate or timing.

    The resulting clip boundaries are timestamps in the actual audio derived
    purely from the word rate, with no audio analysis required.
    """
    if not sentences:
        return []

    word_counts = [max(1, len(s.split())) for s in sentences]
    total_words = sum(word_counts)
    durations   = [total_duration * (wc / total_words) for wc in word_counts]

    segments: List[Dict] = []
    current_text:  List[str]  = []
    current_start: float      = 0.0
    current_dur:   float      = 0.0
    elapsed:       float      = 0.0

    for sentence, dur in zip(sentences, durations):
        if current_text and (current_dur + dur) > max_seconds:
            segments.append({
                "start": current_start,
                "end":   elapsed,
                "text":  " ".join(current_text),
            })
            current_start = elapsed
            current_text  = []
            current_dur   = 0.0

        current_text.append(sentence)
        current_dur += dur
        elapsed     += dur

    if current_text:
        segments.append({
            "start": current_start,
            "end":   total_duration,
            "text":  " ".join(current_text),
        })

    return segments

def segment_long_files(audio_dir: Path, text_dir: Path) -> None:
    """
    For every audio/text pair where the audio exceeds MAX_UTTERANCE_SECONDS:
      1. Split the transcript into sentences.
      2. Assign each sentence a duration proportional to its word count.
      3. Merge sentences into MFA-sized chunks (≤ MAX_UTTERANCE_SECONDS each).
      4. Slice the audio at the computed boundaries using ffmpeg.
      5. Write one .wav + .txt per chunk, then move the originals to data/originals/.
    """
    wav_files  = list(audio_dir.glob("*.wav"))
    long_files = [
        w for w in wav_files
        if (get_audio_duration(w) or 0) > MAX_UTTERANCE_SECONDS
    ]
    if not long_files:
        return

    if not check_ffmpeg():
        print("⚠ ffmpeg is required for auto-segmentation.")
        print("  Install it with:  conda install -c conda-forge ffmpeg")
        sys.exit(1)

    for wav in long_files:
        txt = text_dir / (wav.stem + ".txt")
        if not txt.exists():
            print(f"⚠ No transcript for {wav.name} – cannot segment, skipping.")
            continue

        duration = get_audio_duration(wav) or 0.0
        mins, secs = divmod(int(duration), 60)
        print(f"\n✂ {wav.name} is {mins}m {secs}s – auto-segmenting into short clips...")

        raw_text  = txt.read_text(encoding="utf-8")
        sentences = split_transcript_into_sentences(raw_text)
        if not sentences:
            print(f"  ⚠ Transcript for {wav.name} is empty – skipping.")
            continue

        segments = build_proportional_segments(sentences, duration)
        print(f"  Merged {len(sentences)} sentence(s) into {len(segments)} clip(s) "
              f"(≤{MAX_UTTERANCE_SECONDS}s each).")

        base_stem = re.sub(r'_\d{3}$', '', wav.stem)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            written = 0
            clip_pairs = []

            for i, seg in enumerate(segments, start=1):
                seg_dur = seg["end"] - seg["start"]
                if seg_dur <= 0 or not seg["text"]:
                    continue
                name     = f"{base_stem}_{i:03d}"
                tmp_wav  = tmp_path / f"{name}.wav"
                tmp_txt  = tmp_path / f"{name}.txt"
                r = subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav),
                     "-ss", f"{seg['start']:.3f}", "-t", f"{seg_dur:.3f}",
                     "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(tmp_wav)],
                    capture_output=True, text=True,
                )
                if r.returncode != 0:
                    print(f"  ⚠ Could not extract segment {i}: {r.stderr.strip()[-150:]}")
                    continue
                tmp_txt.write_text(seg["text"], encoding="utf-8")
                clip_pairs.append((tmp_wav, tmp_txt,
                                   audio_dir / f"{name}.wav",
                                   text_dir  / f"{name}.txt"))
                written += 1

            if written == 0:
                print(f"  ⚠ No clips written for {wav.name} – originals kept.")
                continue

            originals_audio = audio_dir.parent / "originals" / "audio"
            originals_text  = audio_dir.parent / "originals" / "text"
            originals_audio.mkdir(parents=True, exist_ok=True)
            originals_text.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wav), originals_audio / wav.name)
            shutil.move(str(txt), originals_text  / txt.name)
            print(f"  Original files moved to data/originals/ for safekeeping.")

            for tmp_wav, tmp_txt, final_wav, final_txt in clip_pairs:
                shutil.move(str(tmp_wav), final_wav)
                shutil.move(str(tmp_txt), final_txt)

        print(f"  ✓ Written {written} clip(s).")

# ── File-pair validation ───────────────────────────────────────────────────────

def validate_file_pairs(audio_dir: Path, text_dir: Path) -> bool:
    audio_stems = {p.stem for p in audio_dir.glob("*.wav")}
    text_stems  = {p.stem for p in text_dir.glob("*.txt")}
    matched     = audio_stems & text_stems

    for stem in sorted(audio_stems - text_stems):
        print(f"⚠ {stem}.wav has no matching transcript – will be skipped by MFA.")
    for stem in sorted(text_stems - audio_stems):
        print(f"⚠ {stem}.txt has no matching audio – will be skipped by MFA.")

    if not matched:
        print("❌ No matched audio/text pairs found.")
        print("   Make sure each audio file has a transcript with the same filename.")
        return False

    print(f"✓ {len(matched)} matched audio/text pair(s) ready for alignment.")
    return True

# ── Main alignment pipeline ────────────────────────────────────────────────────

def run_alignment(overwrite: bool = False) -> None:
    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt  = MODEL_DIR / "twi_lexicon.txt"

    # 1. Convert audio to MFA format
    if AUDIO_DIR.exists():
        convert_audio_to_mfa_format(AUDIO_DIR)

    # 2. Auto-segment any long files
    if AUDIO_DIR.exists() and TEXT_DIR.exists():
        segment_long_files(AUDIO_DIR, TEXT_DIR)

    # 3. Pre-flight checks
    errors = False
    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.glob("*.wav")):
        print("❌ No .wav files found in data/audio/. Please add your audio files.")
        errors = True
    if not TEXT_DIR.exists() or not any(TEXT_DIR.glob("*.txt")):
        print("❌ No .txt files found in data/text/. Please add your transcripts.")
        errors = True
    if not model_zip.exists():
        print("❌ Acoustic model missing. Run with --update to download.")
        errors = True
    if not dict_txt.exists():
        print("❌ Lexicon missing. Run with --update to download.")
        errors = True
    if errors:
        sys.exit(1)

    if not validate_file_pairs(AUDIO_DIR, TEXT_DIR):
        sys.exit(1)

    # 4. Copy .txt files into data/audio/ so MFA finds them next to the .wav files.
    #    MFA's --txt_dir flag is unreliable across versions; co-location always works.
    for txt_file in TEXT_DIR.glob("*.txt"):
        dest = AUDIO_DIR / txt_file.name
        if not dest.exists():
            shutil.copy2(str(txt_file), dest)

    # 5. Run MFA
    cmd = [
        "mfa", "align",
        str(AUDIO_DIR), str(dict_txt), str(model_zip), str(OUTPUT_DIR),
        "--clean",
    ]
    if overwrite:
        cmd.append("--overwrite")

    print("\n🚀 Running alignment...")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Alignment complete! Results saved in {OUTPUT_DIR}/")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Alignment failed: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check the MFA log files shown above for details.")
        print("  2. Make sure all transcript words appear in models/twi_lexicon.txt.")
        print("  3. Run:  mfa validate data/audio models/twi_lexicon.txt models/twi_acoustic_model.zip")
        sys.exit(1)

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Twi Forced Aligner – automatic download, segmentation, and alignment."
    )
    parser.add_argument("--update",    action="store_true", help="Re-download model and dictionary")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not ensure_model_and_dict(REPO, force_update=args.update):
        sys.exit(1)

    run_alignment(overwrite=args.overwrite)

if __name__ == "__main__":
    main()
