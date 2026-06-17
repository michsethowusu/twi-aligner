#!/usr/bin/env python3
"""
Twi grapheme-to-phoneme (G2P) preprocessing for forced alignment.

The acoustic model aligns *phones*, not words. A word can only be aligned if it
appears in the pronunciation dictionary (``models/twi_lexicon.txt``). Words that
are absent are out-of-vocabulary (OOV) and align poorly.

Twi orthography is almost perfectly phonemic, so the pronunciation of a word is
simply its sequence of (lower-cased) characters, each character being one phone.
This module turns arbitrary Twi words into dictionary entries in that exact
format and is used by ``align.py`` to expand the lexicon on the fly so most
input text aligns without any manual dictionary editing.

The acoustic model's phone inventory (see the model's ``meta.json``):
    a b c d e f g h i k l m n o p r s t u w y ɔ ɛ

Output entry format (tab-separated), identical to ``twi_lexicon.txt``:
    word<TAB>p h o n e s

Standalone use:
    # build an expanded lexicon from transcripts in data/text/
    python g2p_twi.py --text-dir data/text --lexicon models/twi_lexicon.txt \
        --merged-out models/twi_lexicon.expanded.txt

    # or generate prons for a plain word list
    python g2p_twi.py --words new_words.txt --lexicon models/twi_lexicon.txt
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Exact phone inventory of the acoustic model (from meta.json["phones"]).
PHONES: Set[str] = set("a b c d e f g h i k l m n o p r s t u w y ɔ ɛ".split())

# Non-phone characters that legitimately occur in spelling and are simply
# dropped when building a pronunciation (clitic markers, hyphens, punctuation).
DROP: Set[str] = set("'’‘`-‑–—/.,?!:;\"“”«»()[]{}…·*")

# Fallback for letters absent from Twi phonology (mostly English loanwords).
# Edit to taste; remove an entry to have such words reported as unmappable.
FALLBACK: Dict[str, str] = {
    "j": "y",     # English ⟨j⟩ ≈ /dʒ/ → closest available
    "v": "f",
    "z": "s",
    "x": "k s",
    "q": "k",
}


def word_to_pron(word: str) -> Tuple[str, Set[str]]:
    """Map one word to (pronunciation, unmapped_chars).

    ``pronunciation`` is a space-separated phone string, or '' if the word
    cannot be represented at all. ``unmapped_chars`` is non-empty when the word
    contains characters with no phone (e.g. digits), in which case the caller
    should treat the word as unmappable rather than emit a partial pronunciation.
    """
    # NFD → strip combining marks (e.g. the tense-vowel mark U+0318 on e̘/i̘/o̘/u̘
    # collapses to the base vowel the model knows) → NFC → lower-case.
    w = unicodedata.normalize("NFD", word)
    w = "".join(c for c in w if not unicodedata.combining(c))
    w = unicodedata.normalize("NFC", w).lower()

    phones: List[str] = []
    unmapped: Set[str] = set()
    for ch in w:
        if ch in PHONES:
            phones.append(ch)
        elif ch in DROP or ch.isspace():
            continue
        elif ch in FALLBACK:
            phones.extend(FALLBACK[ch].split())
        else:
            unmapped.add(ch)
    return " ".join(phones), unmapped


def normalize_key(token: str) -> str:
    """Lower-cased NFC form used both as the dictionary key and for lookup."""
    return unicodedata.normalize("NFC", token).lower()


def _strip_edges(token: str) -> str:
    """Trim leading/trailing punctuation so 'Aane.' and '(obi)' tokenize cleanly."""
    return token.strip("".join(DROP))


def words_from_text_dir(text_dir: Path) -> Set[str]:
    """Collect the unique words appearing across every .txt transcript."""
    words: Set[str] = set()
    for txt in sorted(Path(text_dir).glob("*.txt")):
        for token in txt.read_text(encoding="utf-8").split():
            tok = _strip_edges(token)
            if tok:
                words.add(normalize_key(tok))
    return words


def words_from_file(path: Path) -> Set[str]:
    """Collect unique words from a plain word list (one or many per line)."""
    words: Set[str] = set()
    for token in Path(path).read_text(encoding="utf-8").split():
        tok = _strip_edges(token)
        if tok:
            words.add(normalize_key(tok))
    return words


def load_lexicon(path: Path) -> Dict[str, str]:
    """Load an existing lexicon into {word: pronunciation}."""
    entries: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            key, _, pron = line.partition("\t")
            if not pron:  # tolerate space-separated word/pron split
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, pron = parts
            entries[key] = pron
    return entries


def expand_lexicon(words: Iterable[str], base_lexicon: Path,
                   out_path: Path) -> Dict:
    """Write ``base_lexicon`` plus generated entries for every OOV word.

    Returns a report dict with coverage statistics and the list of words that
    could not be mapped (so the caller can surface/log them).
    """
    base = load_lexicon(base_lexicon)
    have = set(base.keys())

    unique = set(words)
    new_entries: Dict[str, str] = {}
    unmappable: List[Tuple[str, List[str]]] = []

    for w in sorted(unique):
        if w in have:
            continue
        pron, bad = word_to_pron(w)
        if not pron or bad:
            unmappable.append((w, sorted(bad)))
            continue
        new_entries[w] = pron

    merged = dict(base)
    merged.update(new_entries)
    with open(out_path, "w", encoding="utf-8") as f:
        for k in sorted(merged):
            f.write(f"{k}\t{merged[k]}\n")

    covered = len(unique) - len(unmappable)
    coverage = (covered / len(unique) * 100.0) if unique else 100.0
    return {
        "unique_words": len(unique),
        "already_in_lexicon": len(unique & have),
        "generated": len(new_entries),
        "unmappable": unmappable,
        "coverage_pct": coverage,
        "total_entries": len(merged),
        "out_path": str(out_path),
    }


def print_report(report: Dict, unmappable_out: Path = None) -> None:
    """Human-readable coverage summary; optionally dump unmappable words to file."""
    bad = report["unmappable"]
    print(
        f"  Lexicon coverage: {report['coverage_pct']:.1f}% "
        f"({report['unique_words'] - len(bad)}/{report['unique_words']} unique words)"
    )
    print(
        f"    {report['already_in_lexicon']} already in dictionary, "
        f"{report['generated']} generated, {len(bad)} unmappable."
    )
    if bad:
        preview = ", ".join(w for w, _ in bad[:10])
        more = "" if len(bad) <= 10 else f" … (+{len(bad) - 10} more)"
        print(f"    Unmappable (will use the OOV/<unk> model): {preview}{more}")
        if unmappable_out is not None:
            unmappable_out.write_text(
                "\n".join(f"{w}\t{','.join(c)}" for w, c in bad) + "\n",
                encoding="utf-8",
            )
            print(f"    Full list written to {unmappable_out}")


# ── Standalone CLI ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-dir", help="directory of .txt transcripts to scan")
    src.add_argument("--words", help="plain word list (one or many words per line)")
    ap.add_argument("--lexicon", required=True, help="existing base lexicon")
    ap.add_argument("--merged-out", help="write base lexicon + new entries here")
    ap.add_argument("--unmappable-out", help="write the unmappable word list here")
    args = ap.parse_args()

    words = (words_from_text_dir(Path(args.text_dir)) if args.text_dir
             else words_from_file(Path(args.words)))

    out = Path(args.merged_out) if args.merged_out else Path(args.lexicon).with_suffix(".expanded.txt")
    report = expand_lexicon(words, Path(args.lexicon), out)
    print(f"Wrote {report['total_entries']} entries -> {out}", file=sys.stderr)
    print_report(report, Path(args.unmappable_out) if args.unmappable_out else None)


if __name__ == "__main__":
    main()
