"""Download the pre-trained acoustic model and dictionary from GitHub Releases."""
from pathlib import Path
from typing import List, Dict, Optional

import requests
from tqdm import tqdm

from .paths import MODEL_DIR

# Repository whose Releases host the model assets. Change this if you fork.
REPO = "GhanaNLP/twi-aligner"


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


def ensure_model_and_dict(repo: str = REPO, force_update: bool = False) -> bool:
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
