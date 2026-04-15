#!/usr/bin/env python3
"""Download Piper TTS models for all supported languages."""

import os
import sys
import urllib.request

from languages_config import get_language_models

_SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPER_MODEL_DIR = os.path.join(_SOURCE_DIR, "piper_models")

MODELS = get_language_models()


def _piper_download_url(model_name: str) -> str:
    """Derive the HuggingFace download URL from a piper model name."""
    parts = model_name.split("-")
    lang_region = parts[0]
    lang = lang_region.split("_")[0]
    speaker = parts[1]
    quality = parts[2] if len(parts) > 2 else "medium"
    base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    return f"{base}/{lang}/{lang_region}/{speaker}/{quality}/"


def download_model(model_name: str) -> None:
    """Download .onnx and .onnx.json for a model if not already present."""
    os.makedirs(PIPER_MODEL_DIR, exist_ok=True)
    base_url = _piper_download_url(model_name)

    for suffix in (f"{model_name}.onnx", f"{model_name}.onnx.json"):
        dest = os.path.join(PIPER_MODEL_DIR, suffix)
        if os.path.exists(dest):
            print(f"  skip {suffix} (already exists)")
            continue
        url = base_url + suffix
        print(f"  downloading {suffix} ...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  saved {suffix} ({size_mb:.1f} MB)")


def main():
    langs = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())

    for lang in langs:
        if lang not in MODELS:
            print(f"Unknown language: {lang}  (available: {', '.join(MODELS)})")
            sys.exit(1)

    for lang in langs:
        model = MODELS[lang]
        print(f"[{lang}] {model}")
        download_model(model)

    print("Done.")


if __name__ == "__main__":
    main()
