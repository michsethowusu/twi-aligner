#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
push_to_hub.py
==============
Reads the TSV produced by align_dataset.py and pushes to Hugging Face Hub
in the same format as  ghananlpcommunity/asante-twi-yarngpt-aligned.

No local audio files are needed — audio, text, and speaker_id are all
pulled directly from the source HF dataset using the sample index encoded
in each sample_id (sample_00042 → source row 42).

Columns pushed:
    speaker_id  – str    from source dataset (or "speaker_id_9" if unresolved)
    text        – str    from source dataset (or words joined from TSV)
    audio       – Audio  from source dataset, resampled to 16 kHz
    alignment   – str    JSON [{start, end, text, score}, …] from TSV

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
python push_to_hub.py \\
    --tsv            alignments.tsv \\
    --hub-repo       "your-org/your-dataset-name" \\
    --hub-token      hf_... \\
    --source-dataset "Ghana/twi-religious-speech" \\
    --source-split   train \\
    --source-spk-col speaker_id \\
    --source-txt-col sentence \\
    --source-aud-col audio

---------------------------------------------------------------------------
SPEAKER ID / AUDIO DETECTION STRATEGY
---------------------------------------------------------------------------
For every sample_id that matches  sample_NNNNN  the number N is used
directly as the row index into the source dataset — no count comparison
needed.  This means rows skipped during alignment (silence, short audio,
etc.) are handled naturally: the TSV simply won't have those indices and
they are never looked up.

For sample_ids that don't match the pattern (real file stems) we fall back
to stem-matching via --source-id-col.

Anything unresolved gets speaker_id = "speaker_id_9".
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FALLBACK_SPEAKER = "speaker_id_9"
SAMPLE_IDX_RE    = re.compile(r"^sample_(\d+)$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitise_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(raw))


def read_tsv(tsv_path: Path) -> dict:
    """
    Returns {sample_id: [{"word", "start", "end"}, …]} preserving word order.
    """
    utterances: dict = defaultdict(list)
    with open(tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"sample_id", "word", "start_sec", "end_sec"}
        missing  = required - set(reader.fieldnames or [])
        if missing:
            print(f"❌ TSV is missing required column(s): {missing}")
            sys.exit(1)
        for row in reader:
            sid = row["sample_id"].strip()
            if not sid or not row["word"].strip():
                continue
            utterances[sid].append({
                "word":  row["word"].strip(),
                "start": float(row["start_sec"]),
                "end":   float(row["end_sec"]),
            })
    return dict(utterances)


def build_alignment_json(words: list) -> str:
    return json.dumps(
        [{"start": w["start"], "end": w["end"], "text": w["word"], "score": 0.0}
         for w in words],
        ensure_ascii=False,
    )


# ── Source dataset loading ────────────────────────────────────────────────────

def load_source(args):
    try:
        from datasets import load_dataset, Audio as HFAudio
    except ImportError:
        print("❌ 'datasets' not installed.  Run:  pip install datasets")
        sys.exit(1)

    print(f"\n📥 Loading source dataset '{args.source_dataset}' "
          f"(split: {args.source_split})…")
    try:
        ds = load_dataset(
            args.source_dataset,
            split=args.source_split,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ Could not load source dataset: {e}")
        sys.exit(1)

    if args.source_aud_col in ds.column_names:
        ds = ds.cast_column(args.source_aud_col, HFAudio(sampling_rate=16_000))
    else:
        print(f"⚠  Audio column '{args.source_aud_col}' not found "
              f"(available: {ds.column_names}).")

    print(f"  ✓ {len(ds)} row(s) in source dataset.")
    return ds


def build_row_map(sample_ids: list, ds, args) -> dict:
    """
    Returns {sample_id: {"speaker_id", "text", "audio_array", "sampling_rate"}}.

    - If sample_id matches sample_NNNNN  →  use N directly as the row index.
      No count comparison. Works even when alignment skipped some rows.
    - Otherwise  →  stem-match via --source-id-col.
    - Unresolved  →  FALLBACK_SPEAKER, audio_array = None.
    """
    n_source = len(ds)

    # Separate index-style from stem-style ids
    index_ids = {sid: int(SAMPLE_IDX_RE.match(sid).group(1))
                 for sid in sample_ids if SAMPLE_IDX_RE.match(sid)}
    stem_ids  = [sid for sid in sample_ids if sid not in index_ids]

    print(f"\n  Source dataset rows  : {n_source}")
    print(f"  TSV utterances total : {len(sample_ids)}")
    print(f"  Index-style ids      : {len(index_ids)}  (sample_NNNNN → row N)")
    print(f"  Stem-style ids       : {len(stem_ids)}")

    def extract(row) -> dict:
        spk  = str(row.get(args.source_spk_col, FALLBACK_SPEAKER)).strip() \
               or FALLBACK_SPEAKER
        text = str(row.get(args.source_txt_col, "")).strip()
        aud  = row.get(args.source_aud_col)
        return {
            "speaker_id":    spk,
            "text":          text,
            "audio_array":   aud["array"]         if aud else None,
            "sampling_rate": aud["sampling_rate"] if aud else 16_000,
        }

    mapping = {}

    # ── Index-style: direct lookup by row number ──────────────────────────────
    for sid, idx in index_ids.items():
        if idx >= n_source:
            print(f"  ⚠ Index {idx} out of range (dataset has {n_source} rows) "
                  f"— fallback for '{sid}'.")
            mapping[sid] = {"speaker_id": FALLBACK_SPEAKER, "text": "",
                            "audio_array": None, "sampling_rate": 16_000}
        else:
            mapping[sid] = extract(ds[idx])

    # ── Stem-style: match sanitised file stem ─────────────────────────────────
    if stem_ids:
        id_col = args.source_id_col
        if id_col not in ds.column_names:
            print(f"  ⚠ Column '{id_col}' not found for stem-matching. "
                  f"Stem-style ids → '{FALLBACK_SPEAKER}'.")
            for sid in stem_ids:
                mapping[sid] = {"speaker_id": FALLBACK_SPEAKER, "text": "",
                                "audio_array": None, "sampling_rate": 16_000}
        else:
            seen_counts: dict = defaultdict(int)
            stem_map: dict    = {}
            for row in ds:
                raw_id    = row.get(id_col, "")
                stem      = Path(str(raw_id)).stem
                base_id   = sanitise_id(stem)
                count     = seen_counts[base_id]
                mapped_id = base_id if count == 0 else f"{base_id}_{count:04d}"
                seen_counts[base_id] += 1
                stem_map[mapped_id]  = extract(row)

            for sid in stem_ids:
                mapping[sid] = stem_map.get(
                    sid,
                    {"speaker_id": FALLBACK_SPEAKER, "text": "",
                     "audio_array": None, "sampling_rate": 16_000},
                )

    resolved   = sum(1 for v in mapping.values()
                     if v["speaker_id"] != FALLBACK_SPEAKER)
    unresolved = len(mapping) - resolved
    print(f"\n  ✓ Resolved   : {resolved}")
    print(f"  ✗ Unresolved : {unresolved}  (→ '{FALLBACK_SPEAKER}')")
    return mapping


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push aligned TSV + source HF audio to Hub (no local WAVs needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--tsv",       required=True,
                        help="TSV produced by align_dataset.py")
    parser.add_argument("--hub-repo",  required=True,
                        help="HF Hub dataset repo, e.g. your-org/your-dataset")
    parser.add_argument("--hub-token", required=True,
                        help="HF Hub write token (hf_...)")

    parser.add_argument("--source-dataset", required=True,
                        help="Original HF dataset repo to pull audio/text/speaker from")
    parser.add_argument("--source-split",   default="train")
    parser.add_argument("--source-spk-col", default="speaker_id")
    parser.add_argument("--source-txt-col", default="sentence")
    parser.add_argument("--source-aud-col", default="audio")
    parser.add_argument("--source-id-col",  default="path",
                        help="Used only for stem-matching fallback (default: path)")

    parser.add_argument("--split",   default="train")
    parser.add_argument("--private", action="store_true")

    args = parser.parse_args()

    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        print(f"❌ TSV file not found: {tsv_path}")
        sys.exit(1)

    try:
        from datasets import Dataset, Audio as HFAudio
    except ImportError:
        print("❌ 'datasets' not installed.  Run:  pip install datasets")
        sys.exit(1)

    # ── Read TSV ──────────────────────────────────────────────────────────────
    print(f"\n📄 Reading alignments from '{tsv_path}'…")
    utterances = read_tsv(tsv_path)
    print(f"  ✓ {len(utterances)} unique utterance(s) found.")
    if not utterances:
        print("❌ No utterances found in TSV.")
        sys.exit(1)

    # ── Load source dataset & build map ───────────────────────────────────────
    ds      = load_source(args)
    row_map = build_row_map(list(utterances.keys()), ds, args)

    # ── Build rows ────────────────────────────────────────────────────────────
    print("\n⚙️  Building dataset rows…")
    rows          = []
    missing_audio = 0

    for sid, words in utterances.items():
        src         = row_map.get(sid, {})
        audio_array = src.get("audio_array")
        sr          = src.get("sampling_rate", 16_000)

        if audio_array is None:
            print(f"  ⚠ No audio for '{sid}' — skipping.")
            missing_audio += 1
            continue

        speaker_id = src.get("speaker_id", FALLBACK_SPEAKER)
        text       = src.get("text") or " ".join(w["word"] for w in words)
        alignment  = build_alignment_json(words)

        rows.append({
            "speaker_id": speaker_id,
            "text":       text,
            "audio":      {"array": audio_array, "sampling_rate": sr},
            "alignment":  alignment,
        })

    print(f"  ✓ {len(rows)} row(s) ready.  ({missing_audio} skipped — no audio.)")
    if not rows:
        print("❌ No rows to push.")
        sys.exit(1)

    # Preview
    ex = rows[0]
    print(f"\n  Example row:")
    print(f"    speaker_id : {ex['speaker_id']}")
    txt_p = ex['text'][:80] + "…" if len(ex['text']) > 80 else ex['text']
    print(f"    text       : {txt_p}")
    print(f"    audio      : array shape {ex['audio']['array'].shape}, "
          f"{ex['audio']['sampling_rate']} Hz")
    print(f"    alignment  : {ex['alignment'][:120]}…")

    # ── Build & push HF Dataset ───────────────────────────────────────────────
    print("\n📦 Building Hugging Face Dataset…")
    hf_dataset = Dataset.from_list(rows)
    hf_dataset = hf_dataset.cast_column("audio", HFAudio(sampling_rate=16_000))
    print(f"  ✓ Schema: {hf_dataset.features}")

    print(f"\n🚀 Pushing to '{args.hub_repo}' (split: {args.split})…")
    try:
        hf_dataset.push_to_hub(
            args.hub_repo,
            split=args.split,
            token=args.hub_token,
            private=args.private,
        )
    except Exception as e:
        print(f"❌ push_to_hub failed: {e}")
        sys.exit(1)

    print(f"\n✅ Done!  Dataset at: https://huggingface.co/datasets/{args.hub_repo}")


if __name__ == "__main__":
    main()
