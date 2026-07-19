#!/usr/bin/env python3
"""Compare dense and sparse prefix hits for one dataset."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--dense-base-url",
        default="http://127.0.0.1:30001",
    )
    parser.add_argument(
        "--sparse-base-url",
        default="http://127.0.0.1:30000",
    )
    parser.add_argument("--timeout", type=float, default=1_800)
    parser.add_argument("--logprob-atol", type=float, default=0.1)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    prefix_ids = dataset["prefix_input_ids"]
    assert len(prefix_ids) == dataset["prefix_length"]
    assert len(prefix_ids) > 0
    assert len(dataset["suffixes"]) == 2
    assert all(1_000 <= token_id < 30_000 for token_id in prefix_ids)
    for suffix in dataset["suffixes"]:
        assert len(suffix["input_ids"]) == suffix["length"]
        assert suffix["input_ids"]
        assert all(1_000 <= token_id < 30_000 for token_id in suffix["input_ids"])
    return dataset


def request_json(
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
    expect_json: bool = True,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    return json.loads(body) if expect_json and body else {}


def check_health(base_url: str, timeout: float) -> None:
    request_json(
        "GET",
        f"{base_url}/health",
        timeout=timeout,
        expect_json=False,
    )


def flush_cache(base_url: str, timeout: float) -> None:
    request_json(
        "POST",
        f"{base_url}/flush_cache?timeout=120",
        timeout=timeout,
        expect_json=False,
    )


def generate(
    base_url: str,
    input_ids: list[int],
    *,
    max_new_tokens: int,
    timeout: float,
    logprob_start_len: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if logprob_start_len is not None:
        payload.update(
            {
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": logprob_start_len,
            }
        )
    response = request_json(
        "POST",
        f"{base_url}/generate",
        timeout=timeout,
        payload=payload,
    )
    if "meta_info" not in response:
        raise RuntimeError(f"invalid /generate response: {response}")
    return response


def normalize_logprobs(items: list[Any]) -> list[tuple[float, int]]:
    values = []
    for item in items:
        if item is None or item[0] is None:
            continue
        value = float(item[0])
        token_id = int(item[1])
        if not math.isfinite(value):
            raise AssertionError(f"non-finite logprob: {item}")
        values.append((value, token_id))
    return values


def compare_logprobs(
    dense: dict[str, Any],
    sparse: dict[str, Any],
    *,
    atol: float,
) -> float:
    dense_meta = dense["meta_info"]
    sparse_meta = sparse["meta_info"]
    fields = ("input_token_logprobs", "output_token_logprobs")
    max_abs_diff = 0.0
    max_diff_detail = None
    for field in fields:
        dense_values = normalize_logprobs(dense_meta[field])
        sparse_values = normalize_logprobs(sparse_meta[field])
        assert len(dense_values) == len(sparse_values), (
            field,
            len(dense_values),
            len(sparse_values),
        )
        assert dense_values, f"no comparable values in {field}"
        pairs = zip(dense_values, sparse_values, strict=True)
        for index, (
            (dense_value, dense_token),
            (sparse_value, sparse_token),
        ) in enumerate(pairs):
            assert dense_token == sparse_token, (
                field,
                index,
                dense_token,
                sparse_token,
            )
            abs_diff = abs(dense_value - sparse_value)
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
                max_diff_detail = (
                    field,
                    index,
                    dense_token,
                    dense_value,
                    sparse_value,
                )
    assert max_abs_diff <= atol, (max_abs_diff, atol, max_diff_detail)
    return max_abs_diff


def run_prefix_hit(
    *,
    base_url: str,
    prefix_ids: list[int],
    full_input_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    prefix_length = len(prefix_ids)
    flush_cache(base_url, timeout)
    warm = generate(
        base_url,
        prefix_ids,
        max_new_tokens=0,
        timeout=timeout,
    )
    assert warm["meta_info"]["cached_tokens"] == 0

    hit = generate(
        base_url,
        full_input_ids,
        max_new_tokens=1,
        timeout=timeout,
        logprob_start_len=prefix_length,
    )
    assert hit["meta_info"]["cached_tokens"] == prefix_length, (
        hit["meta_info"]["cached_tokens"],
        prefix_length,
    )
    return hit


def run_suffix_case(
    dataset: dict[str, Any],
    suffix: dict[str, Any],
    *,
    dense_base_url: str,
    sparse_base_url: str,
    timeout: float,
    logprob_atol: float,
) -> None:
    prefix_ids = dataset["prefix_input_ids"]
    suffix_ids = suffix["input_ids"]
    prefix_length = len(prefix_ids)
    full_input_ids = prefix_ids + suffix_ids
    label = f'{dataset["name"]}/{suffix["name"]}'

    print(f"[{label}] dense warm + prefix hit")
    dense = run_prefix_hit(
        base_url=dense_base_url,
        prefix_ids=prefix_ids,
        full_input_ids=full_input_ids,
        timeout=timeout,
    )

    print(f"[{label}] sparse warm + prefix hit")
    sparse = run_prefix_hit(
        base_url=sparse_base_url,
        prefix_ids=prefix_ids,
        full_input_ids=full_input_ids,
        timeout=timeout,
    )
    assert dense["output_ids"] == sparse["output_ids"]
    max_abs_diff = compare_logprobs(dense, sparse, atol=logprob_atol)
    print(
        f"[{label}] PASS cached_tokens={prefix_length} "
        f"max_abs_logprob_diff={max_abs_diff:.6g}"
    )
    flush_cache(dense_base_url, timeout)
    flush_cache(sparse_base_url, timeout)


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    print(
        f'validated {dataset["name"]}: prefix={dataset["prefix_length"]}, '
        f'suffixes={[suffix["length"] for suffix in dataset["suffixes"]]}'
    )
    if args.validate_only:
        return

    dense_base_url = args.dense_base_url.rstrip("/")
    sparse_base_url = args.sparse_base_url.rstrip("/")
    assert dense_base_url != sparse_base_url
    check_health(dense_base_url, args.timeout)
    check_health(sparse_base_url, args.timeout)
    for suffix in dataset["suffixes"]:
        run_suffix_case(
            dataset,
            suffix,
            dense_base_url=dense_base_url,
            sparse_base_url=sparse_base_url,
            timeout=args.timeout,
            logprob_atol=args.logprob_atol,
        )


if __name__ == "__main__":
    main()
