from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def _collect_safetensor_state(weights_root: Path, patterns: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """
    Load safetensor shards under `weights_root` into a single state dict.

    Parameters
    ----------
    weights_root:
        Path containing the downloaded snapshot with safetensors files.
    patterns:
        Optional whitelist of glob patterns (relative to weights_root) that
        restrict which safetensors files are loaded.
    """
    weights_root = weights_root.resolve()
    if not weights_root.exists():
        raise FileNotFoundError(f"Weights directory does not exist: {weights_root}")

    tensor_paths = set()
    if patterns:
        for pattern in patterns:
            tensor_paths.update(weights_root.glob(pattern))
    else:
        tensor_paths.update(weights_root.glob("**/*.safetensors"))

    tensor_paths = sorted(p for p in tensor_paths if p.is_file())
    if not tensor_paths:
        raise FileNotFoundError(f"No .safetensors files found under {weights_root}")

    logger.info("Loading %d safetensor shard(s)...", len(tensor_paths))
    state_dict: Dict[str, torch.Tensor] = {}
    for path in tensor_paths:
        shard = load_file(str(path), device="cpu")
        overlap = set(state_dict).intersection(shard)
        if overlap:
            raise ValueError(f"Duplicate tensor keys detected when loading {path}: {sorted(overlap)[:3]}")
        state_dict.update(shard)
    logger.info("Loaded %d tensors from safetensors.", len(state_dict))
    return state_dict


class _ContainerModule(nn.Module):
    """Simple container module to help build a nested module hierarchy."""

    def __init__(self) -> None:
        super().__init__()


class AlphaOnlyModel(nn.Module):
    """
    Minimal module hierarchy populated with nn.Linear layers based on a state dict.

    The goal is to mirror the module naming of the original model sufficiently
    for alpha computation via `compute_alpha_values`.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], target_dtype: Optional[torch.dtype] = None) -> None:
        super().__init__()
        self._build_from_state(state_dict, target_dtype=target_dtype)

    def _build_from_state(self, state_dict: Dict[str, torch.Tensor], target_dtype: Optional[torch.dtype]) -> None:
        bias_lookup = {key[:-5]: tensor for key, tensor in state_dict.items() if key.endswith(".bias")}

        for full_key, weight in state_dict.items():
            if not full_key.endswith(".weight"):
                continue
            if weight.ndim < 2:
                # Skip layer norms / embeddings / scalars; only Linear layers are needed.
                continue

            module_path = full_key[:-7].split(".")
            bias_tensor = bias_lookup.get(full_key[:-7])

            weight_2d = weight.reshape(weight.shape[0], -1)
            in_features = weight_2d.shape[1]
            out_features = weight_2d.shape[0]

            linear = nn.Linear(in_features, out_features, bias=bias_tensor is not None)
            with torch.no_grad():
                linear.weight.copy_(weight_2d.to(dtype=target_dtype) if target_dtype else weight_2d)
                if bias_tensor is not None:
                    linear.bias.copy_(bias_tensor.to(dtype=target_dtype) if target_dtype else bias_tensor)

            parent_module, leaf_name = self._get_parent_container(module_path)
            parent_module.add_module(leaf_name, linear)

    def _get_parent_container(self, path_parts: Iterable[str]) -> tuple[nn.Module, str]:
        parts = list(path_parts)
        if not parts:
            raise ValueError("Module path cannot be empty.")
        parent = self
        for part in parts[:-1]:
            if not hasattr(parent, part):
                container = _ContainerModule()
                parent.add_module(part, container)
            candidate = getattr(parent, part)
            if not isinstance(candidate, nn.Module):
                raise TypeError(f"Encountered non-module attribute '{part}' while building module tree.")
            parent = candidate
        return parent, parts[-1]


def build_delta_net_alpha_model(
    weights_root: Path,
    *,
    target_dtype: Optional[torch.dtype] = None,
    patterns: Optional[Iterable[str]] = None,
) -> nn.Module:
    """
    Construct an AlphaOnlyModel from safetensors weights rooted at `weights_root`.

    Parameters
    ----------
    weights_root:
        Directory containing safetensors weight shards.
    target_dtype:
        Optional dtype to cast the loaded tensors to.
    patterns:
        Optional glob patterns (relative to weights_root) to restrict which files are loaded.
    """
    state_dict = _collect_safetensor_state(weights_root, patterns=patterns)
    model = AlphaOnlyModel(state_dict, target_dtype=target_dtype)
    model.eval()
    return model


