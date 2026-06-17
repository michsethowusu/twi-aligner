# Twi Forced Aligner

A one‑stop tool to align Twi (Akan) audio with text transcripts using a pre‑trained acoustic model. Simply place your files in the `data/` folders and run `python align.py` – everything else is handled automatically.

> **⚠ Domain notice:** The current pre‑trained model was trained exclusively on **religious speech** (Bible readings and sermons). It works well out of the box for similar material, but will produce lower‑quality alignments on conversational Twi, broadcast speech, storytelling, or other domains. If your data comes from a different domain, **we strongly recommend finetuning the model on a sample of your own data** before running full alignment. See [Finetuning](#-finetuning-the-model) below.

---

## ✨ Features

- **Precise word-level alignments** – as a forced aligner, this tool produces exact start/end timestamps for every word, not just utterance boundaries. This goes beyond what CTC-based ASR systems (e.g. wav2vec 2.0, Whisper) provide: CTC models are optimised for transcription and their frame-level posteriors yield imprecise or approximate word boundaries, especially for short function words and consonant clusters common in Twi. Forced alignment with a GMM-HMM acoustic model gives sub-50 ms accuracy at the word level, which is essential for phonetic research, TTS data preparation, and corpus annotation.
- **Tiny model size** – about **80 MB** and runs entirely on CPU.
- **No manual model download** – fetches the acoustic model and dictionary from GitHub Releases automatically.
- **Any audio length** – long recordings are automatically segmented into short clips before alignment; no manual splitting needed.
- **Any audio format** – `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` are all accepted and converted to the correct format automatically.
- **Out-of-vocabulary words handled automatically** – words not in the dictionary are converted to phones on the fly using grapheme-to-phoneme (G2P) rules for Twi, so you are no longer limited to the ~21k words in the bundled lexicon. See [Out-of-Vocabulary Words](#-out-of-vocabulary-oov-words).
- **Caches downloaded files** – subsequent runs are instant.
- Comes with sample audio/text to test the pipeline.

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/GhanaNLP/twi-aligner.git
   cd twi-aligner
   ```

2. **Create the conda environment**

   ```bash
   conda create -n aligner -c conda-forge montreal-forced-aligner ffmpeg
   conda activate aligner
   ```

   > This installs MFA and `ffmpeg` together. Using conda avoids common compilation issues (e.g. `_kalpy` not found) that occur when installing MFA via pip.

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the aligner**

   ```bash
   python align.py
   ```

   - If no model is found locally, you will be prompted to choose a release from GitHub.
   - The model and dictionary are downloaded into `models/`.
   - Audio is converted and segmented automatically if needed.
   - Results appear in `output/` as `.TextGrid` files.

5. **Use your own data**

   - Place your audio files in `data/audio/` and transcripts in `data/text/`.
   - Each audio file needs a matching `.txt` with the same filename (e.g. `speech01.wav` ↔ `speech01.txt`).
   - For long recordings, place the full transcript in the `.txt` file — the script automatically splits it into sentences and segments the audio to match. For best results use one sentence per line, but a plain paragraph works too.
   - Run `python align.py` again.

---

## 🎯 Finetuning the Model

The base acoustic model was trained on religious speech. If you are working with a different domain, adapting the model to your data will noticeably improve alignment quality.

### How it works

MFA acoustic models are Kaldi-based **GMM-HMM** models. Finetuning (adaptation) runs **MAP (Maximum A Posteriori) adaptation** and **fMLLR (feature-space MLLR)** on your labelled data. This updates the Gaussian mixture parameters to fit your speakers and domain without discarding what the base model already learned — similar in spirit to transfer learning for neural models.

### What you need

- At least **15–30 minutes** of transcribed Twi audio from your target domain. More data gives better results; 1–2 hours is ideal.
- The same conda environment used for `align.py`.
- The base model already downloaded (run `python align.py --update` if not).

### Data layout

```
data/finetune/
    audio/   ← your .wav / .mp3 / .flac / .m4a files
    text/    ← matching .txt transcripts, one per audio file
```

The transcript format is the same as for alignment: UTF-8, filename matching the audio file, one sentence per line.

### Run adaptation

```bash
python finetune.py
```

This will:
1. Convert audio to 16 kHz mono WAV.
2. Validate audio/transcript pairs.
3. Warn if there is less than 15 minutes of audio.
4. Run `mfa adapt` against the base model.
5. Save the adapted model to `models/twi_acoustic_model_adapted.zip`.

### Options

```bash
python finetune.py --data-dir my_recordings/        # custom data directory
python finetune.py --output-model twi_conv_model    # custom output name
python finetune.py --num-jobs 4                     # parallelise (speeds things up)
python finetune.py --overwrite                      # replace existing adapted model
```

### Using the adapted model

After adaptation, swap in the new model before running alignment:

```bash
cp models/twi_acoustic_model_adapted.zip models/twi_acoustic_model.zip
python align.py
```

Or run MFA directly:

```bash
mfa align data/audio/ models/twi_lexicon.txt models/twi_acoustic_model_adapted.zip output/
```

---

## 🔤 Out-of-Vocabulary (OOV) Words

A forced aligner can only place a word if it knows how that word is pronounced. The bundled dictionary (`models/twi_lexicon.txt`) covers ~21,000 Twi words, but real transcripts always contain words it has never seen — names, loanwords, inflected forms, and so on.

`align.py` handles these automatically. **Before alignment runs**, it scans your transcripts, finds every word that is missing from the dictionary, and generates a pronunciation for it using grapheme-to-phoneme (G2P) rules. The generated entries are merged into a working copy of the lexicon (`models/twi_lexicon.expanded.txt`) — your original dictionary is never modified.

### How it works

Twi orthography is almost perfectly phonemic: a word's pronunciation is essentially its sequence of characters. The G2P step lower-cases each word, drops punctuation and clitic markers, maps a few non-Twi letters found in loanwords (`j→y`, `v→f`, `z→s`, `x→k s`, `q→k`), and emits one phone per remaining character — using **only** the 23 phones the acoustic model was trained on (`a b c d e f g h i k l m n o p r s t u w y ɔ ɛ`). In practice this covers the large majority of any Twi transcript.

### Coverage report

Each run prints a coverage summary, for example:

```
🔤 Expanding lexicon for out-of-vocabulary words...
  Lexicon coverage: 96.2% (25/26 unique words)
    18 already in dictionary, 7 generated, 1 unmappable.
    Unmappable (will use the OOV/<unk> model): 99bottles
    Full list written to output/oov_unmappable.txt
```

Words that cannot be represented with the model's phones (e.g. anything containing digits, or symbols like `%`) are listed in `output/oov_unmappable.txt`. These are passed to MFA's standard OOV handling and will not get precise alignments — typically a small fraction of a transcript. To fix them, either rewrite them in normal Twi spelling (e.g. spell out numbers) or add a manual pronunciation to `models/twi_lexicon.txt`.

### Building a lexicon yourself

You can also run the G2P step standalone — handy for inspecting what will be generated, or for preparing a dictionary outside the aligner:

```bash
# from a directory of transcripts
python g2p_twi.py --text-dir data/text --lexicon models/twi_lexicon.txt \
    --merged-out models/twi_lexicon.expanded.txt --unmappable-out output/oov_unmappable.txt

# or from a plain word list (one or many words per line)
python g2p_twi.py --words new_words.txt --lexicon models/twi_lexicon.txt
```

### Disabling G2P

To align strictly against the bundled dictionary (treating every other word as OOV, the old behaviour), pass `--no-g2p`:

```bash
python align.py --no-g2p
```

---

## 🔧 Advanced Options

- `--update` – Force re‑download of the model/dictionary.

  ```bash
  python align.py --update
  ```

- `--overwrite` – Overwrite existing alignment files in `output/`.

  ```bash
  python align.py --overwrite
  ```

- `--no-g2p` – Disable automatic G2P expansion and align only against the bundled dictionary.

  ```bash
  python align.py --no-g2p
  ```

---

## 🗂 Aligning a Dataset

`align_dataset.py` is a convenience wrapper that handles bulk alignment from two common input sources, then writes a TSV of word-level timestamps ready for downstream use.

### Install extra dependencies

```bash
pip install datasets soundfile TextGrid   # TextGrid is optional but recommended
```

### Option A – Hugging Face dataset

```bash
python align_dataset.py \
    --dataset Ghana/twi-religious-speech \
    --split train \
    --audio-col audio \
    --text-col transcription \
    --output-tsv alignments.tsv
```

### Option B – Local CSV/TSV + audio files

Your metadata file needs at least two columns: one for audio paths (absolute or relative to the CSV) and one for transcripts.

```
path,sentence
recordings/001.wav,meda wo ase
recordings/002.mp3,ɛte sɛn
```

```bash
python align_dataset.py \
    --csv metadata.csv \
    --audio-col path \
    --text-col sentence \
    --output-tsv alignments.tsv
```

### Output

Both modes produce the same TSV format:

| sample_id | word | start_sec | end_sec | duration_sec |
|-----------|------|-----------|---------|--------------|
| sample_00001 | meda | 0.1200 | 0.3800 | 0.2600 |
| sample_00001 | wo | 0.3800 | 0.5400 | 0.1600 |

### Common options

```bash
--max-samples 50     # process only the first N samples (useful for testing)
--overwrite          # overwrite existing alignment files in output/
--keep-data          # keep prepared files in data/audio/ and data/text/
```

---

## 📁 Data Format

- **Audio**: Any common format (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`). Converted to 16 kHz mono WAV automatically.
- **Transcripts**: UTF‑8 `.txt` files. The filename must match the audio file. For long recordings, the full transcript goes in a single `.txt` — one sentence per line gives the best segmentation, but a plain paragraph is also handled automatically.
- **Dictionary**: Downloaded automatically as part of the model release. Words not in the lexicon are pronounced automatically via G2P (see [Out-of-Vocabulary Words](#-out-of-vocabulary-oov-words)); you only need to edit `models/twi_lexicon.txt` for words the G2P step reports as unmappable.

---

## ❓ FAQ

**Q: Why use this instead of a CTC-based aligner like wav2vec 2.0 or Whisper?**  
A: CTC models are optimised for transcription accuracy. Their internal frame-level scores can be used to estimate word boundaries, but the estimates are often imprecise — particularly for short words, unstressed syllables, and the consonant clusters common in Twi. A forced aligner with a GMM-HMM model is purpose-built for boundary detection and routinely achieves sub-50 ms word-level accuracy. If you need timestamps for phonetic research, TTS data curation, or fine-grained corpus annotation, forced alignment is the right tool.

**Q: The script says "No releases found".**  
A: Make sure you are using the correct repository name. If you forked this repo, update the `REPO` variable at the top of `align.py`.

**Q: Can I use a locally trained model?**  
A: Yes. Place your model zip and dictionary in `models/` named `twi_acoustic_model.zip` and `twi_lexicon.txt`. The download step will be skipped.

**Q: Alignment quality is poor on my data.**  
A: The base model was trained on religious speech. If your audio comes from a different domain, finetune the model on a sample of your own data — see [Finetuning](#-finetuning-the-model) above.

**Q: Alignment is slow.**  
A: Alignment time scales with the amount of audio. Increase parallel jobs by adding `--num_jobs 4` to the MFA command in `align.py`, or pass `--num-jobs 4` to `finetune.py`.

**Q: I get an error about `_kalpy` missing.**  
A: MFA was likely installed with `pip`. Reinstall using conda as shown in the Quick Start – it handles all native dependencies correctly.

**Q: A word in my transcript is not in the dictionary.**  
A: It is handled automatically — `align.py` generates a pronunciation for it via G2P before alignment (see [Out-of-Vocabulary Words](#-out-of-vocabulary-oov-words)). Only words reported in `output/oov_unmappable.txt` (e.g. those containing digits or symbols) need attention: rewrite them in normal Twi spelling, or add a manual entry to `models/twi_lexicon.txt` using the format `word p h o n e m e s`.

---

## 📦 Included Sample Data

- `data/audio/sample1.wav` – Short utterance "meda wo ase"
- `data/text/sample1.txt` – "meda wo ase"

Use these to verify everything works before processing your own files.

---

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- The acoustic model was trained using the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) on a corpus of Twi religious speech.
- Thanks to all contributors.

---

**Happy aligning!**  
If you encounter issues, please open an issue on GitHub.
