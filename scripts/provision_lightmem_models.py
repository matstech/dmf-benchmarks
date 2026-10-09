"""Explicitly download or verify the pinned local LightMem model files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen


PROFILE = Path(__file__).resolve().parents[1] / "config/lightmem/lightmem-local-v1.json"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Parent directory for the two model folders")
    parser.add_argument("--download", action="store_true", help="Explicitly permit Hugging Face downloads")
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    for section, folder in (("embedding", "all-MiniLM-L6-v2"), ("segmenter", "llmlingua-2")):
        model = profile[section]
        repository, revision = model["repository"], model["revision"]
        if len(revision) != 40 or not all(char in "0123456789abcdef" for char in revision):
            raise SystemExit(f"Invalid pinned revision for {section}")
        for name, expected in model["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"Invalid model path: {name}")
            target = args.output / folder / relative
            if target.is_file() and digest_file(target) == expected:
                print(f"verified {section}/{name}")
                continue
            if not args.download:
                raise SystemExit(f"Missing or mismatched {target}; rerun with --download to provision explicitly")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
            try:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as destination:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        destination.write(chunk)
                if digest_file(temporary) != expected:
                    raise ValueError(f"SHA-256 mismatch for {section}/{name}")
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"downloaded and verified {section}/{name}")


if __name__ == "__main__":
    main()
