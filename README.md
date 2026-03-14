# Twi Forced Aligner

A one‑stop tool to align Twi (Akan) audio with text transcripts using a pre‑trained acoustic model. Simply place your files in the `data/` folders and run `python align.py` – everything else is handled automatically.

## ✨ Features

- **Tiny model size** – about **80 MB** and runs entirely on CPU.
- **No manual model download** – fetches the acoustic model and dictionary from GitHub Releases automatically.
- **Any audio length** – long recordings are automatically segmented into short clips before alignment; no manual splitting needed.
- **Any audio format** – `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg` are all accepted and converted to the correct format automatically.
- **Caches downloaded files** – subsequent runs are instant.
- Comes with sample audio/text to test the pipeline.

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

## 🔧 Advanced Options

- `--update` – Force re‑download of the model/dictionary.

  ```bash
  python align.py --update
  ```

- `--overwrite` – Overwrite existing alignment files in `output/`.

  ```bash
  python align.py --overwrite
  ```

- `--no-tones` – Omit tone markers from auto-generated pronunciations.

  ```bash
  python align.py --no-tones
  ```

## 📁 Data Format

- **Audio**: Any common format (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`). Converted to 16 kHz mono WAV automatically.
- **Transcripts**: UTF‑8 `.txt` files. The filename must match the audio file. For long recordings, the full transcript goes in a single `.txt` — one sentence per line gives the best segmentation, but a plain paragraph is also handled automatically.
- **Dictionary**: Downloaded automatically. Missing words are added using the built‑in G2P converter.

## ❓ FAQ

**Q: The script says "No releases found".**  
A: Make sure you are using the correct repository name. If you forked this repo, update the `REPO` variable at the top of `align.py`.

**Q: Can I use a locally trained model?**  
A: Yes. Place your model zip and dictionary in `models/` named `twi_acoustic_model.zip` and `twi_lexicon.txt`. The download step will be skipped.

**Q: Alignment is slow.**  
A: Alignment time scales with the amount of audio. For large corpora, increase parallel jobs by adding `--num_jobs 4` to the MFA command in `align.py`.

**Q: I get an error about `_kalpy` missing.**  
A: MFA was likely installed with `pip`. Reinstall using conda as shown in the Quick Start – it handles all native dependencies correctly.

**Q: The dictionary doesn't contain all my words.**  
A: Missing words are added automatically via the G2P converter. If G2P isn't installed, add them manually to `models/twi_lexicon.txt` in the format `word p h o n e m e s`.

## 📦 Included Sample Data

- `data/audio/sample1.wav` – Short utterance "meda wo ase"
- `data/text/sample1.txt` – "meda wo ase"

Use these to verify everything works before processing your own files.

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgements

- The acoustic model was trained using the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) on a corpus of Twi speech.
- Thanks to all contributors.

---

**Happy aligning!**  
If you encounter issues, please open an issue on GitHub.
