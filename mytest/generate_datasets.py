#!/usr/bin/env python3
"""Generate deterministic token-id datasets for sparse-prefix correctness tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PREFIX_SPECS = (
    ("1k", 1_024),
    ("2k", 2_048),
    ("4k", 4_096),
    ("10k", 10_240),
    ("16k", 16_384),
)
TOKEN_ID_MIN = 1_000
TOKEN_ID_MAX_EXCLUSIVE = 30_000
DEFAULT_SUFFIX_LENGTH = 32
BASE_SEED = 0x5A17_2026


def generate_token_ids(length: int, seed: int) -> list[int]:
    """Return a portable deterministic sequence in the configured token range."""

    span = TOKEN_ID_MAX_EXCLUSIVE - TOKEN_ID_MIN
    state = seed & 0x7FFF_FFFF
    token_ids = []
    for _ in range(length):
        state = (1_103_515_245 * state + 12_345) & 0x7FFF_FFFF
        token_ids.append(TOKEN_ID_MIN + state % span)
    return token_ids


def build_dataset(label: str, prefix_length: int, suffix_length: int, index: int):
    prefix_seed = BASE_SEED + index * 10_000 + 1
    prefix_input_ids = generate_token_ids(prefix_length, prefix_seed)
    suffixes = []
    for suffix_index in range(2):
        suffix_seed = BASE_SEED + index * 10_000 + 101 + suffix_index
        suffixes.append(
            {
                "name": f"suffix_{suffix_index + 1}",
                "length": suffix_length,
                "input_ids": generate_token_ids(suffix_length, suffix_seed),
            }
        )

    assert len(prefix_input_ids) == prefix_length
    assert len(suffixes) == 2
    assert suffixes[0]["input_ids"] != suffixes[1]["input_ids"]
    return {
        "schema_version": 1,
        "name": f"prefix_{label}",
        "prefix_length": prefix_length,
        "prefix_input_ids": prefix_input_ids,
        "suffixes": suffixes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets",
    )
    parser.add_argument(
        "--suffix-length",
        type=int,
        default=DEFAULT_SUFFIX_LENGTH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suffix_length <= 0:
        raise ValueError("suffix length must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    for index, (label, prefix_length) in enumerate(PREFIX_SPECS):
        dataset = build_dataset(label, prefix_length, args.suffix_length, index)
        file_name = f"prefix_{label}.json"
        payload = json.dumps(dataset, separators=(",", ":")) + "\n"
        (args.output_dir / file_name).write_text(payload, encoding="utf-8")
        manifest_entries.append(
            {
                "file": file_name,
                "prefix_length": prefix_length,
                "suffix_count": 2,
                "suffix_lengths": [args.suffix_length, args.suffix_length],
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
        )

    manifest = {
        "schema_version": 1,
        "length_unit": "tokens",
        "k_definition": 1_024,
        "token_id_range": [TOKEN_ID_MIN, TOKEN_ID_MAX_EXCLUSIVE],
        "generator_seed": BASE_SEED,
        "datasets": manifest_entries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
