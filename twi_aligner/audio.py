"""Audio conversion and automatic segmentation of long recordings."""
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

from .paths import MAX_UTTERANCE_SECONDS


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
