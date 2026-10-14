"""Download only the pinned official BEAM Parquet shards for materialization."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from materialize_beam import DATASET_REVISIONS, SOURCE_HASHES, sha256, source_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".dmf-bench/upstream"))
    args = parser.parse_args()
    root = args.output.resolve()
    for name, revision in DATASET_REVISIONS.items():
        snapshot_download(
            repo_id=f"Mohammadta/{name}", repo_type="dataset", revision=revision,
            local_dir=root / f"{name}-dataset", allow_patterns=["data/*.parquet", "README.md"],
        )
    for relative, path in source_files(root):
        if not path.is_file() or sha256(path) != SOURCE_HASHES[relative]:
            raise ValueError(f"BEAM source digest mismatch: {relative}")
    print(f"Verified {len(SOURCE_HASHES)} pinned BEAM Parquet shards in {root}")


if __name__ == "__main__":
    main()
