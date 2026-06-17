#!/usr/bin/env python3
"""
Twi Forced Aligner – downloads a pre-trained acoustic model from GitHub Releases
(if not already present) and aligns audio/text using MFA.

Long recordings are automatically segmented into short utterance-level clips
using proportional word-rate splitting before alignment, so you can hand in
any length of audio. Words missing from the dictionary are pronounced
automatically via grapheme-to-phoneme conversion (see twi_aligner.g2p).

Run from the repository root:  python align.py
"""
import sys
import shutil
import argparse
import subprocess

from . import g2p
from .paths import MODEL_DIR, AUDIO_DIR, TEXT_DIR, OUTPUT_DIR, MAX_UTTERANCE_SECONDS
from .download import REPO, ensure_model_and_dict
from .audio import convert_audio_to_mfa_format, segment_long_files


# ── File-pair validation ───────────────────────────────────────────────────────

def validate_file_pairs(audio_dir, text_dir) -> bool:
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


# ── Lexicon expansion (out-of-vocabulary handling) ───────────────────────────────

def expand_lexicon_for_transcripts(dict_txt):
    """Generate pronunciations for any OOV word in data/text/ and return the
    path to an expanded lexicon. Falls back to the base lexicon on any error."""
    try:
        words = g2p.words_from_text_dir(TEXT_DIR)
        if not words:
            return dict_txt
        expanded = MODEL_DIR / "twi_lexicon.expanded.txt"
        print("\n🔤 Expanding lexicon for out-of-vocabulary words...")
        report = g2p.expand_lexicon(words, dict_txt, expanded)
        g2p.print_report(report, OUTPUT_DIR / "oov_unmappable.txt")
        return expanded
    except Exception as e:
        print(f"⚠ G2P expansion failed ({e}); using base lexicon as-is.")
        return dict_txt


# ── Main alignment pipeline ────────────────────────────────────────────────────

def run_alignment(overwrite: bool = False, use_g2p: bool = True,
                  max_seconds: float = MAX_UTTERANCE_SECONDS,
                  beam: int = None, retry_beam: int = None) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt  = MODEL_DIR / "twi_lexicon.txt"

    # 1. Convert audio to MFA format
    if AUDIO_DIR.exists():
        convert_audio_to_mfa_format(AUDIO_DIR)

    # 2. Auto-segment any long files
    if AUDIO_DIR.exists() and TEXT_DIR.exists():
        segment_long_files(AUDIO_DIR, TEXT_DIR, max_seconds)

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

    # 5. Expand the lexicon so out-of-vocabulary words can still be aligned.
    align_dict = expand_lexicon_for_transcripts(dict_txt) if use_g2p else dict_txt

    # 6. Run MFA
    cmd = [
        "mfa", "align",
        str(AUDIO_DIR), str(align_dict), str(model_zip), str(OUTPUT_DIR),
        "--clean",
    ]
    if beam is not None:
        cmd += ["--beam", str(beam)]
    if retry_beam is not None:
        cmd += ["--retry_beam", str(retry_beam)]
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
        print("  2. Words that could not be auto-mapped are listed in output/oov_unmappable.txt;")
        print("     add manual pronunciations for them to models/twi_lexicon.txt if needed.")
        print("  3. Run:  mfa validate data/audio models/twi_lexicon.txt models/twi_acoustic_model.zip")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Twi Forced Aligner – automatic download, segmentation, and alignment."
    )
    parser.add_argument("--update",    action="store_true", help="Re-download model and dictionary")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--no-g2p",    action="store_true",
                        help="Disable automatic G2P lexicon expansion for OOV words")
    parser.add_argument("--max-seconds", type=float, default=MAX_UTTERANCE_SECONDS,
                        help=f"Max clip length before auto-segmentation, in seconds "
                             f"(default: {MAX_UTTERANCE_SECONDS}). Lower it if long "
                             f"utterances fail to align.")
    parser.add_argument("--beam", type=int, default=None,
                        help="MFA alignment beam (default: MFA's own, 10). Raise it "
                             "(e.g. 100) when utterances fail to align, which is common "
                             "for audio outside the model's training domain.")
    parser.add_argument("--retry-beam", type=int, default=None,
                        help="MFA retry beam for utterances that fail the first pass "
                             "(default: MFA's own, 40). Try ~4x --beam, e.g. 400.")
    args = parser.parse_args()

    if not ensure_model_and_dict(REPO, force_update=args.update):
        sys.exit(1)

    run_alignment(overwrite=args.overwrite, use_g2p=not args.no_g2p,
                  max_seconds=args.max_seconds,
                  beam=args.beam, retry_beam=args.retry_beam)


if __name__ == "__main__":
    main()
