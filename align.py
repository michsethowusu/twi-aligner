#!/usr/bin/env python3
"""
Twi Forced Aligner – downloads a pre-trained acoustic model from GitHub Releases
(if not already present) and aligns audio/text using MFA.
"""

import os
import sys
import subprocess
import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

import requests
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration – change if you fork the repository
REPO = "yourusername/twi-aligner"          # GitHub repository
MODEL_DIR = Path("models")
AUDIO_DIR = Path("data/audio")
TEXT_DIR = Path("data/text")
OUTPUT_DIR = Path("output")
# ----------------------------------------------------------------------

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def get_latest_release_info(repo: str) -> Optional[Dict]:
    """Fetch the latest release from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching latest release: {e}")
        return None

def get_all_releases(repo: str) -> List[Dict]:
    """Fetch all releases (paginated)."""
    releases = []
    page = 1
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
    """Download a file with a progress bar."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        desc=desc or dest.name,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def select_release_interactive(releases: List[Dict]) -> Optional[Dict]:
    """Let the user choose a release from a list."""
    if not releases:
        return None
    if len(releases) == 1:
        return releases[0]
    print("\nMultiple model releases found. Please choose one:")
    for i, rel in enumerate(releases, 1):
        name = rel.get('name') or rel.get('tag_name')
        published = rel.get('published_at', '')[:10]
        print(f"  {i}. {name} ({published})")
    while True:
        try:
            choice = int(input("Enter number (or 0 to cancel): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(releases):
                return releases[choice-1]
        except ValueError:
            pass
        print("Invalid choice, try again.")

def ensure_model_and_dict(repo: str, force_update: bool = False) -> bool:
    """
    Make sure we have both acoustic model zip and dictionary.
    Returns True if successful, False otherwise.
    """
    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt = MODEL_DIR / "twi_lexicon.txt"

    # If both exist and not forcing update, just return success
    if model_zip.exists() and dict_txt.exists() and not force_update:
        print("✓ Model and dictionary already present (use --update to re-download).")
        return True

    # Fetch all releases
    print("Fetching available model releases from GitHub...")
    releases = get_all_releases(repo)
    if not releases:
        print("❌ No releases found. Please check the repository name or network.")
        return False

    # Let user pick one (or use latest if only one)
    selected = select_release_interactive(releases)
    if not selected:
        print("Download cancelled.")
        return False

    tag = selected['tag_name']
    print(f"Selected release: {tag}")

    # Find assets: we need twi_acoustic_model.zip and twi_lexicon.txt
    assets = {asset['name']: asset['browser_download_url'] for asset in selected['assets']}
    needed = ['twi_acoustic_model.zip', 'twi_lexicon.txt']
    for name in needed:
        if name not in assets:
            print(f"❌ Required asset '{name}' not found in release {tag}.")
            return False

    # Download model zip
    print(f"Downloading {needed[0]}...")
    download_file(assets[needed[0]], model_zip, desc="Model ZIP")
    # Download dictionary
    print(f"Downloading {needed[1]}...")
    download_file(assets[needed[1]], dict_txt, desc="Dictionary")

    print("✓ Model and dictionary downloaded successfully.")
    return True

def run_alignment(overwrite: bool = False) -> None:
    """Run MFA alignment with the downloaded model/dict."""
    model_zip = MODEL_DIR / "twi_acoustic_model.zip"
    dict_txt = MODEL_DIR / "twi_lexicon.txt"

    # Sanity checks
    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.glob("*.wav")):
        print("⚠ Warning: No .wav files found in data/audio/. Please add your audio.")
    if not TEXT_DIR.exists() or not any(TEXT_DIR.glob("*.txt")):
        print("⚠ Warning: No .txt files found in data/text/. Please add your transcripts.")
    if not model_zip.exists():
        print("❌ Acoustic model missing. Run with --update to download.")
        sys.exit(1)
    if not dict_txt.exists():
        print("❌ Dictionary missing. Run with --update to download.")
        sys.exit(1)

    cmd = [
        "mfa", "align",
        str(AUDIO_DIR),
        str(TEXT_DIR),
        str(model_zip),
        str(dict_txt),
        str(OUTPUT_DIR),
    ]
    if overwrite:
        cmd.append("--overwrite")

    print("\n🚀 Running alignment...")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Alignment complete! Results saved in {OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Alignment failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Twi Forced Aligner with automatic model download")
    parser.add_argument("--update", action="store_true", help="Force re-download of model/dictionary (even if already present)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    # Ensure we have the model/dict (download if needed or forced)
    if not ensure_model_and_dict(REPO, force_update=args.update):
        sys.exit(1)

    # Run alignment
    run_alignment(overwrite=args.overwrite)

if __name__ == "__main__":
    main()
