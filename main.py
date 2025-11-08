import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

from alphalab.utils_alpha import compute_alpha_values, save_alpha_to_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Hill alpha values for every linear layer of a model."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model path or Hugging Face model identifier (e.g., 'mistralai/Mixtral-8x7B-v0.1').",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("alpha_values.csv"),
        help="Destination CSV file to store alpha values (default: ./alpha_values.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for caching alpha computations.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["baseline", "farms"],
        default="farms",
        help=(
            "Alpha computation mode. "
            "'baseline' forces full SVD, "
            "'farms' forces FARMS sampling."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        help="Transformer attention implementation to request from transformers (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype used when loading the model weights (default: float16).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging output (only warnings and above).",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_model(model_name_or_path: str, attn_impl: str, dtype: torch.dtype):
    logger.info(f"Loading model: {model_name_or_path}")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_impl,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    logger.info("Model loaded successfully.")
    return model


def _determine_use_farms(alpha_mode: str) -> Optional[bool]:
    if alpha_mode == "baseline":
        return False
    if alpha_mode == "farms":
        return True
    return None


def main() -> int:
    args = _parse_args()

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    dtype = _resolve_dtype(args.dtype)
    try:
        model = _load_model(args.model, args.attn_implementation, dtype)
    except Exception as exc:
        logger.error(f"Failed to load model '{args.model}': {exc}")
        return 1

    use_farms = _determine_use_farms(args.alpha_mode)

    cache_dir: Optional[str] = None
    if args.cache_dir is not None:
        cache_dir = str(args.cache_dir.resolve())

    logger.info("Computing alpha values...")
    with torch.inference_mode():
        alpha_values = compute_alpha_values(
            model,
            cache_dir=cache_dir,
            use_farms=use_farms,
        )

    logger.info(f"Computed alpha for {len(alpha_values)} layers.")

    output_path = args.output_csv.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_alpha_to_csv(alpha_values, str(output_path))
    logger.info(f"Alpha values saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

