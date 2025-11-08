#!/usr/bin/env python
"""
Workaround script to compute alpha values for `fla-hub/delta_net-1.3B-100B`
using the existing AlphaLab utilities (`alphalab.utils_alpha`), without relying
on flash-linear-attention.

This script downloads the model snapshot locally (if needed), loads it through
`transformers`, and then delegates alpha computation to `compute_alpha_values`
followed by persisting the results via `save_alpha_to_csv`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download

from alphalab.utils_alpha import compute_alpha_values, save_alpha_to_csv
from models.delta_net import build_delta_net_alpha_model


log = logging.getLogger("workaround")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download delta_net weights and compute alpha values using AlphaLab utilities."
    )
    parser.add_argument(
        "--repo-id",
        default="fla-hub/delta_net-1.3B-100B",
        help="Hugging Face repo id to download (default: %(default)s)",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("cache/delta_net_weights"),
        help="Local directory to store or locate the snapshot (default: ./cache/delta_net_weights).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("delta_net_alpha_values.csv"),
        help="Destination CSV file for computed alpha values.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/alpha"),
        help="Cache directory to reuse previously computed alpha values (default: ./cache/alpha).",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["baseline", "farms"],
        default="farms",
        help="Alpha computation mode passed through to AlphaLab utils (default: farms).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype used when instantiating linear layers (default: float16).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force re-download of the snapshot even if it already exists locally.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational log output.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _download_snapshot(repo_id: str, local_dir: Path, force: bool) -> Path:
    log.info("Ensuring snapshot is available locally...")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=not force,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.py", "*.bin"],
    )
    log.info("Snapshot ready at %s", snapshot_path)
    return Path(snapshot_path)


def _determine_use_farms(alpha_mode: str) -> Optional[bool]:
    if alpha_mode == "baseline":
        return False
    if alpha_mode == "farms":
        return True
    return None


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    dtype = _resolve_dtype(args.dtype)

    try:
        snapshot_path = _download_snapshot(args.repo_id, args.local_dir, args.force_redownload)
    except Exception as exc:
        log.error("Failed to download snapshot: %s", exc)
        return 1

    try:
        model = build_delta_net_alpha_model(snapshot_path, target_dtype=dtype)
    except Exception as exc:
        log.error("Failed to construct model from weights: %s", exc)
        return 1

    cache_dir: Optional[str] = str(args.cache_dir.resolve()) if args.cache_dir else None
    use_farms = _determine_use_farms(args.alpha_mode)

    log.info("Computing alpha values...")
    with torch.inference_mode():
        alpha_values = compute_alpha_values(
            model,
            cache_dir=cache_dir,
            use_farms=use_farms,
        )

    log.info("Computed alpha for %d layers.", len(alpha_values))

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_alpha_to_csv(alpha_values, str(output_csv))
    log.info("Alpha values saved to %s", output_csv)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)



