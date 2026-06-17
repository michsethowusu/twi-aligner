# Twi Forced Aligner

Get precise, word-level timestamps for Twi (Akan) audio from its transcript. Drop your audio and text into `data/`, run one command, and get a `.TextGrid` with the start and end time of every word.

```bash
python align.py
```

That's the whole workflow. The script downloads the model, converts and segments your audio, fills in pronunciations for any word, runs the alignment, and writes results to `output/`.

---

## ✨ What you get

- **Word-level timestamps** – exact start/end times for every word, accurate to ~50 ms. Forced alignment is purpose-built for this and is far more precise than word boundaries estimated from CTC models like wav2vec 2.0 or Whisper.
- **Any word** – words outside the bundled dictionary are pronounced automatically via [grapheme-to-phoneme conversion](#-any-word-out-of-vocabulary-handling). You are not limited to a fixed vocabulary.
- **Any audio format or length** – `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` are converted automatically, and long recordings are split into clips for you.
- **Runs on CPU** – the model is ~80 MB; no GPU required.
- **Zero setup after install** – the model and dictionary download on first run and are cached afterwards.

---

## 🚀 Quick Start

**1. Clone and enter the repo**

```bash
git clone https://github.com/GhanaNLP/twi-aligner.git
cd twi-aligner
```

**2. Create the environment** (conda is strongly recommended — it installs MFA and ffmpeg with the right native libraries)

```bash
conda create -n aligner -c conda-forge montreal-forced-aligner ffmpeg
conda activate aligner
pip install -r requirements.txt
```

**3. Try it on the included sample**

```bash
python align.py
```

On first run you'll be asked to pick a model release; it downloads into `models/`. When it finishes, open `output/sample1.TextGrid` (e.g. in [Praat](https://www.praat.org/)) to see the aligned words. That's it — you're ready to use your own audio.

---

## 🎧 Aligning your own audio

The aligner processes everything in `data/`. Each audio file needs a transcript with the **same filename**:

```
data/
  audio/   speech01.wav      meeting.mp3
  text/    speech01.txt      meeting.txt
```

Then run:

```bash
python align.py
```

Results land in `output/` as one `.TextGrid` per input file.

### Examples

**A single short clip**

```bash
# data/audio/greeting.wav  +  data/text/greeting.txt  (contents: "meda wo ase")
python align.py
# → output/greeting.TextGrid
```

**A long recording** — just give the whole transcript; the audio is segmented automatically.

```bash
# data/audio/sermon.mp3  (20 minutes)
# data/text/sermon.txt   (the full transcript — one sentence per line works best)
python align.py
# → output/sermon_001.TextGrid, sermon_002.TextGrid, ...
```

**Many files at once** — drop as many matched pairs into `data/audio` and `data/text` as you like and run `python align.py` once; they're all aligned in a single pass.

> **Tip:** one sentence per line gives the cleanest segmentation for long files, but a plain paragraph is handled too.

---

## 📤 Understanding the output

Each `.TextGrid` contains a **words** tier and a **phones** tier with interval boundaries you can open in Praat, ELAN, or parse programmatically. For bulk work, [`align_dataset.py`](#-batch-aligning-a-dataset) can flatten alignments straight into a TSV:

| sample_id | word | start_sec | end_sec | duration_sec |
|-----------|------|-----------|---------|--------------|
| sample_00001 | meda | 0.1200 | 0.3800 | 0.2600 |
| sample_00001 | wo | 0.3800 | 0.5400 | 0.1600 |

---

## 🔤 Any word: out-of-vocabulary handling

A forced aligner can only place a word it knows the pronunciation of. The bundled dictionary covers ~21,000 Twi words, but real transcripts always contain new ones — names, loanwords, inflected forms. **You don't need to do anything about this.**

Before each run, `align.py` scans your transcripts, generates a pronunciation for every unknown word (Twi spelling is essentially phonemic, so this is highly accurate), and aligns against the expanded dictionary. Your original `models/twi_lexicon.txt` is never modified. You'll see a coverage line like:

```
🔤 Expanding lexicon for out-of-vocabulary words...
  Lexicon coverage: 96.2% (25/26 unique words)
    18 already in dictionary, 7 generated, 1 unmappable.
```

The rare word that can't be mapped (anything with digits or symbols like `%`) is listed in `output/oov_unmappable.txt`. Fix it by rewriting it in normal Twi spelling (e.g. spell numbers out) or adding a line to `models/twi_lexicon.txt` in the form `word p h o n e s`.

<details>
<summary>Generate a dictionary manually, or disable G2P</summary>

Build an expanded lexicon yourself (handy for inspection or offline prep):

```bash
# from your transcripts
python -m twi_aligner.g2p --text-dir data/text --lexicon models/twi_lexicon.txt \
    --merged-out models/twi_lexicon.expanded.txt --unmappable-out output/oov_unmappable.txt

# or from a plain word list
python -m twi_aligner.g2p --words new_words.txt --lexicon models/twi_lexicon.txt
```

To align strictly against the bundled dictionary instead, pass `--no-g2p`.
</details>

---

## ⚙️ Command reference

```bash
python align.py                    # align everything in data/ (robust defaults)
python align.py --overwrite        # re-align, replacing existing output files
python align.py --update           # force re-download of the model and dictionary
python align.py --no-g2p           # align only against the bundled dictionary (no auto-pronunciation)
python align.py --max-seconds 8    # auto-segment clips longer than 8s (default: 10)
python align.py --beam 10 --retry-beam 40   # narrower/faster search for clean in-domain audio
```

**Using your own model:** place a `twi_acoustic_model.zip` and `twi_lexicon.txt` in `models/` and the download step is skipped.

### Robust by default

The defaults are tuned to *just align*, even on audio that differs from the model's religious-speech training:

- **Short clips** — long files are auto-segmented to ≤10s (`--max-seconds`) at sentence boundaries, splitting mid-sentence only when a single sentence is itself too long. Alignment is all-or-nothing per clip, so shorter is more robust.
- **Wide search** — the MFA beam defaults to 100 / 400 (well above MFA's own 10 / 40), which reliably fits out-of-domain audio at the cost of some speed.

These defaults trade a little speed for reliability. On clean, in-domain audio you can speed runs up with a narrower search (`--beam 10 --retry-beam 40`) and/or longer clips (`--max-seconds 20`). If alignment still fails on a whole out-of-domain dataset, [finetuning](#-finetuning-optional) is the more permanent fix.

---

## 🗂 Batch-aligning a dataset

`align_dataset.py` aligns from a Hugging Face dataset or a local CSV/TSV and writes a single TSV of word timestamps.

```bash
pip install datasets soundfile TextGrid
```

**From a Hugging Face dataset:**

```bash
python align_dataset.py \
    --dataset Ghana/twi-religious-speech --split train \
    --audio-col audio --text-col transcription \
    --output-tsv alignments.tsv
```

**From a local CSV** (needs an audio-path column and a transcript column):

```bash
python align_dataset.py \
    --csv metadata.csv --audio-col path --text-col sentence \
    --output-tsv alignments.tsv
```

Useful flags: `--max-samples N` (test on the first N rows), `--overwrite`, `--keep-data`.

---

## 📁 Data format

- **Audio** – any of `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`; converted to 16 kHz mono WAV automatically.
- **Transcripts** – UTF-8 `.txt`, filename matching the audio file. One sentence per line is best for long files; a paragraph also works.
- **Dictionary** – downloaded with the model. Unknown words are pronounced automatically (see [above](#-any-word-out-of-vocabulary-handling)); you only ever edit it for words reported as unmappable.

---

## ❓ FAQ

**Why this instead of wav2vec 2.0 or Whisper?**
Those are built for transcription; their word-boundary estimates are imprecise, especially for short words and the consonant clusters common in Twi. A GMM-HMM forced aligner is purpose-built for boundaries and reaches ~50 ms accuracy — what you want for phonetic research, TTS data prep, and corpus annotation.

**Alignment quality is poor, or I get `NoAlignmentsError`.**
The defaults already use short clips and a wide search (see [Robust by default](#robust-by-default)), so most audio aligns out of the box. If a clip still fails, try an even shorter `--max-seconds 6`. The bundled model was trained on religious speech, so it's most precise on similar material; for a consistently different style (conversational, broadcast, health) adapting the model is the more permanent fix — see [Finetuning](#-finetuning-optional).

**The script says "No releases found".**
Check the repo name; if you forked, update the `REPO` variable at the top of `twi_aligner/download.py`.

**Alignment is slow.**
Time scales with audio length. Add `--num_jobs 4` to the MFA command in `twi_aligner/align.py` to parallelise.

**I get a `_kalpy` missing error.**
MFA was installed via pip. Reinstall with conda as shown in the Quick Start.

---

## 🎯 Finetuning (optional)

> Only needed if alignment boundaries are consistently off because your audio is a very different style from the religious-speech the model was trained on. Most users can skip this.

The model is a Kaldi GMM-HMM; `finetune.py` adapts it to your speakers/domain (MAP + fMLLR) without retraining from scratch. Put ~15–30 min of transcribed audio in `data/finetune/audio` and `data/finetune/text`, then:

```bash
python finetune.py            # writes models/twi_acoustic_model_adapted.zip
```

Swap it in and align as usual:

```bash
cp models/twi_acoustic_model_adapted.zip models/twi_acoustic_model.zip
python align.py
```

Options: `--data-dir`, `--output-model`, `--num-jobs N`, `--overwrite`.

---

## 🧩 Repository layout

```
align.py  align_dataset.py  finetune.py   # thin entry points (run these)
twi_aligner/                              # the package
├── align.py        alignment pipeline
├── dataset.py      batch alignment (HF / CSV)
├── finetune.py     model adaptation
├── g2p.py          grapheme-to-phoneme conversion
├── download.py     model/dictionary download
├── audio.py        conversion + auto-segmentation
└── paths.py        shared data/ models/ output/ locations
data/   models/   output/                 # your inputs and results (gitignored)
```

Installing is optional — `python align.py` works straight from a clone. If you prefer, `pip install -e .` also exposes `twi-align`, `twi-align-dataset`, `twi-finetune`, and `twi-g2p` commands.

---

## 📜 License & credits

[MIT License](LICENSE). Built on the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/); the acoustic model was trained on a corpus of Twi religious speech.

**Happy aligning!** Found a bug or have a question? Please open an issue.
