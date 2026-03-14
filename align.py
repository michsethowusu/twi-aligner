#!/usr/bin/env python3
"""
Twi Forced Aligner – downloads a pre-trained acoustic model from GitHub Releases
(if not already present) and aligns audio/text using MFA.

Long recordings are automatically segmented into short utterance-level clips
using aeneas before alignment, so you can hand in any length of audio.

If words in the transcripts are missing from the lexicon, they are automatically
added using the twi-g2p grapheme-to-phoneme converter.
"""

import re
import sys
import shutil
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Set
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

try:
    from twi_g2p import TwiG2P
    TWIG2P_AVAILABLE = True
except ImportError:
    TWIG2P_AVAILABLE = False

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

    # Compute proportional duration for every sentence
    word_counts = [max(1, len(s.split())) for s in sentences]
    total_words = sum(word_counts)
    durations   = [total_duration * (wc / total_words) for wc in word_counts]

    # Greedily merge sentences into chunks that fit within max_seconds
    segments: List[Dict] = []
    current_text:  List[str]  = []
    current_start: float      = 0.0
    current_dur:   float      = 0.0
    elapsed:       float      = 0.0

    for sentence, dur in zip(sentences, durations):
        # If adding this sentence would breach the limit, flush current chunk
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

    # Flush any remaining text
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
      1. Split the transcript into sentences (handles paragraphs automatically).
      2. Assign each sentence a duration proportional to its word count.
      3. Merge sentences into MFA-sized chunks (≤ MAX_UTTERANCE_SECONDS each).
      4. Slice the audio at the computed boundaries using ffmpeg.
      5. Write one .wav + .txt per chunk, then remove the originals.
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

        # Strip any trailing _NNN suffix from a previous partial run so clip
        # names never snowball (e.g. sample1_001 → sample1_001_001_001…)
        base_stem = re.sub(r'_\d{3}$', '', wav.stem)

        # ── Write clips to a temp dir first to avoid naming collisions ───
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            written = 0
            clip_pairs = []  # (tmp_wav, tmp_txt, final_wav, final_txt)

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

            # Move originals to data/originals/ instead of deleting them,
            # so the user can recover them if needed.
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

# ── Lexicon expansion ──────────────────────────────────────────────────────────

def get_model_phones(model_zip: Path) -> Set[str]:
    """
    Extract the set of phones the acoustic model was trained on from phones.txt
    inside the model zip. Returns an empty set if the file can't be read.
    """
    import zipfile
    phones: Set[str] = set()
    try:
        with zipfile.ZipFile(model_zip) as z:
            # phones.txt is at <model_name>/phones.txt
            phones_file = next(n for n in z.namelist() if n.endswith("phones.txt"))
            for line in z.read(phones_file).decode().splitlines():
                parts = line.strip().split()
                if parts and parts[0] not in ("<eps>", "sil", "spn", "#0", "#1"):
                    phones.add(parts[0])
    except Exception as e:
        print(f"  ⚠ Could not read phones from model: {e}")
    return phones

def normalize_phone(phone: str, valid_phones: Set[str]) -> Optional[str]:
    """
    Map a G2P output phone to one the acoustic model knows.

    Resolution order:
      1. Phone is already valid — use as-is.
      2. Strip Unicode combining diacritics (ATR marks, tone marks) via NFD
         decomposition and try the base character.
      3. For palatalized/labialized digraphs (kʲ kʷ gʲ nʲ hʲ nʷ ŋʷ hʷ etc.)
         keep only the leading base consonant.
      4. Map ŋ → n (ŋ is not in the Twi acoustic model phone set).
      5. Return None if no valid mapping found — phone will be skipped.
    """
    import unicodedata

    if phone in valid_phones:
        return phone

    # Strip combining diacritics (ATR dots, tone marks — Unicode category Mn)
    base = "".join(c for c in unicodedata.normalize("NFD", phone)
                   if unicodedata.category(c) != "Mn")
    if base in valid_phones:
        return base

    # Handle palatalized / labialized digraphs: kʲ kʷ gʲ nʲ hʲ nʷ ŋʷ hʷ …
    # These are single Unicode characters composed of a base + modifier letter.
    # Strip modifier letters (category Lm) to get the base consonant.
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone)
                       if unicodedata.category(c) not in ("Mn", "Lm"))
    if stripped in valid_phones:
        return stripped

    # X-SAMPA style multi-char tokens like k_w, k_j, n_j, N_w — take first char
    if "_" in phone:
        first = phone.split("_")[0]
        # N (eng in X-SAMPA) → n
        first = "n" if first == "N" else first
        if first in valid_phones:
            return first

    # ŋ (eng) → n
    if phone in ("ŋ", "N"):
        return "n" if "n" in valid_phones else None

    # Multi-character clusters (e.g. 'nt', 'mb', 'ng') — split into individual
    # characters after stripping diacritics and return the joined valid phones.
    # We return only the first valid phone here; the caller should use
    # normalize_cluster() for full expansion.
    if len(stripped) > 1:
        valid_chars = [c for c in stripped if c in valid_phones]
        if valid_chars:
            return valid_chars[0]  # caller handles full expansion via normalize_cluster

    # Last resort: take just the first character after NFD stripping
    if stripped and stripped[0] in valid_phones:
        return stripped[0]

    return None

def normalize_cluster(phone: str, valid_phones: Set[str]) -> List[str]:
    """
    Like normalize_phone but returns ALL valid phones from a cluster token.
    e.g. 'nt' → ['n', 't'], 'mb' → ['m', 'b'], 'kʷ' → ['k']
    Use this when building the lexicon so no phones are silently dropped.
    """
    import unicodedata

    # Single phone — delegate to normalize_phone
    single = normalize_phone(phone, valid_phones)

    # Strip diacritics to get base characters
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone)
                       if unicodedata.category(c) not in ("Mn", "Lm"))

    # If it's a multi-char cluster, expand each character individually
    if len(stripped) > 1:
        result = [c for c in stripped if c in valid_phones]
        if result:
            return result

    return [single] if single else []

def build_lexicon_from_g2p(lexicon_path: Path, text_dir: Path,
                            model_zip: Path) -> None:
    """
    Build a lexicon file from scratch using twi-g2p for every unique word
    found across all transcripts, filtering phones to only those present in
    the acoustic model.

    This replaces the downloaded lexicon entirely, ensuring the phone set
    is always compatible with the model.
    """
    if not TWIG2P_AVAILABLE:
        print("⚠ twi-g2p not installed – cannot build lexicon from G2P.")
        print("  Install it with:  pip install git+https://github.com/GhanaNLP/twi-g2p.git")
        print("  Falling back to downloaded lexicon (may cause phone set mismatches).")
        return

    # Get the phone set the model actually knows
    valid_phones = get_model_phones(model_zip)
    if not valid_phones:
        print("⚠ Could not determine model phone set – skipping G2P lexicon build.")
        return
    print(f"  Acoustic model uses {len(valid_phones)} phones: {' '.join(sorted(valid_phones))}")

    # Collect every unique word from all transcripts
    all_words: Set[str] = set()
    for txt_file in text_dir.glob("*.txt"):
        for word in txt_file.read_text(encoding="utf-8").split():
            all_words.add(word)

    if not all_words:
        return

    print(f"  Generating pronunciations for {len(all_words)} unique word(s)...")
    # Always exclude tones – the model phone set has no tone markers.
    # Use vits format: plain space-separated tokens, no curly braces.
    g2p = TwiG2P(output_format="vits", include_tones=False)

    written, skipped = 0, 0
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for word in sorted(all_words):
            try:
                raw_phones = g2p.convert(word).split()
                mapped = []
                for p in raw_phones:
                    mapped.extend(normalize_cluster(p, valid_phones))
                if not mapped:
                    skipped += 1
                    continue
                f.write(f"{word} {' '.join(mapped)}\n")
                written += 1
            except Exception as e:
                skipped += 1

    print(f"✓ Lexicon built: {written} word(s) written, {skipped} skipped (no valid phones).")
    if skipped:
        print(f"  Skipped words will be treated as OOV by MFA and excluded from alignment.")

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

def run_alignment(overwrite: bool = False, no_tones: bool = False) -> None:
    model_zip    = MODEL_DIR / "twi_acoustic_model.zip"
    g2p_lexicon  = MODEL_DIR / "twi_g2p_lexicon.txt"   # built fresh from G2P
    fallback_lex = MODEL_DIR / "twi_lexicon.txt"        # downloaded, used only if G2P unavailable

    # 1. Convert audio to MFA format
    if AUDIO_DIR.exists():
        convert_audio_to_mfa_format(AUDIO_DIR)

    # 2. Auto-segment any long files
    if AUDIO_DIR.exists() and TEXT_DIR.exists():
        segment_long_files(AUDIO_DIR, TEXT_DIR)

    # 3. Build lexicon from G2P, filtered to the model's phone set
    print("\n📖 Building lexicon from G2P...")
    build_lexicon_from_g2p(g2p_lexicon, TEXT_DIR, model_zip)
    dict_txt = g2p_lexicon if g2p_lexicon.exists() else fallback_lex

    # 4. Pre-flight checks
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
        print("❌ No lexicon available. Install twi-g2p or run with --update.")
        errors = True
    if errors:
        sys.exit(1)

    if not validate_file_pairs(AUDIO_DIR, TEXT_DIR):
        sys.exit(1)

    # 5. Copy .txt files into data/audio/ so MFA finds them next to the .wav files.
    #    MFA's --txt_dir flag is unreliable across versions; co-location always works.
    for txt_file in TEXT_DIR.glob("*.txt"):
        dest = AUDIO_DIR / txt_file.name
        if not dest.exists():
            shutil.copy2(str(txt_file), dest)

    # 6. Run MFA
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
