# Twi Forced Aligner

A one‑stop tool to align Twi (Akan) audio with text transcripts using a pre‑trained acoustic model. Simply place your files in the `data/` folders and run `python align.py` – the script automatically downloads the latest (or a chosen) model from GitHub Releases and performs the alignment.

## ✨ Features

- **Tiny model size** – about **80 MB** and runs entirely on CPU.
- **No manual model download** – script fetches the acoustic model and dictionary from GitHub Releases.
- If multiple model versions are released, you can interactively choose which one to use.
- **Caches downloaded files** – subsequent runs are instant.
- Comes with sample audio/text to test the pipeline.

## 📋 Prerequisites

- **Montreal Forced Aligner** installed. The easiest and most reliable way is via **conda** (avoids compilation issues like `_kalpy` not found):
  ```bash
  conda create -n aligner -c conda-forge montreal-forced-aligner
  conda activate aligner
  ```
  See the [official installation guide](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) for alternatives.
- Python 3.8+ and the packages in `requirements.txt`.

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/GhanaNLP/twi-aligner.git
   cd twi-aligner
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the aligner**

   ```bash
   python align.py
   ```

   - If no model is found locally, you will be prompted to choose a release from GitHub.
   - The model and dictionary are downloaded into the `models/` folder.
   - Alignment runs using the sample files in `data/audio` and `data/text`.
   - Results appear in the `output/` folder as `.TextGrid` files.

4. **Use your own data**

   - Replace the sample `.wav` files in `data/audio` with your own (must be 16 kHz mono).
   - Replace the sample `.txt` files in `data/text` with matching transcripts (UTF‑8, same basename).
   - Run `python align.py` again – it will reuse the already downloaded model.

## 🧠 How It Works

- The script contacts the GitHub API to list releases of this repository.
- If multiple releases exist, you choose one interactively.
- It downloads `twi_acoustic_model.zip` and `twi_lexicon.txt` from the selected release assets.
- Then it calls `mfa align` with the provided audio, text, model, and dictionary.
- Alignment is performed and saved in `output/`.

## 🔧 Advanced Options

- `--update` – Force re‑download of the model/dictionary (useful when a new release is available).

  ```bash
  python align.py --update
  ```

- `--overwrite` – Overwrite existing alignment files in `output/`.

  ```bash
  python align.py --overwrite
  ```

## 📁 Data Preparation

- **Audio**: Place `.wav` files in `data/audio/`. They should be mono, 16‑bit PCM, 16 kHz sample rate (MFA’s preferred format). If your audio is different, convert it first (e.g., with `ffmpeg`).
- **Text**: Place `.txt` files in `data/text/`. Each file should contain exactly one line of orthographic transcription (no punctuation needed). The filename must match the audio file (e.g., `speech01.wav` ↔ `speech01.txt`).
- **Dictionary**: The repository includes a default Twi lexicon (`models/twi_lexicon.txt` after download). If you need a custom dictionary, place it in `data/dictionary/` and modify the alignment command manually (advanced users).

## ❓ FAQ

**Q: The script says "No releases found".**  
A: Make sure you are using the correct repository name. If you forked this repo, edit the `REPO` variable at the top of `align.py` to match your username and repository.

**Q: Can I use a locally trained model?**  
A: Yes. Place your model zip and dictionary in the `models/` folder with the exact names `twi_acoustic_model.zip` and `twi_lexicon.txt`. The script will skip the download step.

**Q: Alignment is slow.**  
A: Alignment time depends on the amount of audio. For large corpora, you can increase the number of parallel jobs by adding `--num_jobs 4` to the MFA command inside `align.py` (or use the `-j` option).

**Q: The dictionary doesn't contain all my words.**  
A: Add missing words to `models/twi_lexicon.txt` following the format `word p h o n e m e s`. You can use a phone set consistent with the acoustic model.

**Q: I get an error about `_kalpy` missing.**  
A: This usually means MFA was installed with `pip` but the required C extensions weren’t compiled. Reinstall MFA using **conda** as shown in the prerequisites section – it handles all dependencies correctly.

## 📦 Included Sample Data

- `data/audio/sample1.wav` – Short utterance "meda wo ase"  
- `data/audio/sample2.wav` – Another utterance  
- `data/text/sample1.txt` – "meda wo ase"  
- `data/text/sample2.txt` – corresponding transcript  

Use these to verify everything works before processing your own files.

## 📜 License

[MIT License](LICENSE) 

## 🙏 Acknowledgements

- The acoustic model was trained using the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) on a corpus of Twi speech.
- Thanks to all contributors.

---

**Happy aligning!**  
If you encounter issues, please open an issue on GitHub.
