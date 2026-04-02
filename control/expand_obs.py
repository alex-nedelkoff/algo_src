"""Expand observation input dimension of an MoE checkpoint.

Zero-pads the first linear layer of all experts, router, and value_net
to accommodate new observation dimensions. Existing weights are preserved;
new input columns are initialized to zero so the policy's behavior is
unchanged until fine-tuning.

Usage:
    python -m control.expand_obs \
        --checkpoint outputs/2026-04-01/23-05-22/final_model.zip \
        --old-obs-dim 28 --new-obs-dim 33 \
        --output outputs/expanded_33dim/model.zip
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_policy_state_dict(checkpoint_path: str) -> dict:
    """Load the policy state dict from an SB3 checkpoint .zip."""
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        with zf.open("policy.pth") as f:
            return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)


def expand_input_dim(
    state_dict: dict,
    old_dim: int,
    new_dim: int,
) -> dict:
    """Expand the input dimension of all first-layer weights.

    Identifies layers whose weight shape is (*, old_dim) and expands
    them to (*, new_dim) by zero-padding the new columns.
    """
    new_sd = {}
    n_expanded = 0

    for key, val in state_dict.items():
        if val.dim() == 2 and val.shape[1] == old_dim:
            expanded = torch.zeros(val.shape[0], new_dim, dtype=val.dtype)
            expanded[:, :old_dim] = val
            new_sd[key] = expanded
            n_expanded += 1
            log.info("Expanded %s: %s -> %s", key, val.shape, expanded.shape)
        else:
            new_sd[key] = val.clone()

    log.info("Expanded %d layers from %d -> %d input dims", n_expanded, old_dim, new_dim)
    return new_sd


def save_expanded_checkpoint(
    original_path: str,
    output_path: str,
    new_policy_sd: dict,
) -> None:
    """Save expanded checkpoint by replacing policy.pth in the zip."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(original_path, "r") as zf_in:
        names = zf_in.namelist()
        contents = {}
        for name in names:
            contents[name] = zf_in.read(name)

    buf = io.BytesIO()
    torch.save(new_policy_sd, buf)
    contents["policy.pth"] = buf.getvalue()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
        for name, data in contents.items():
            zf_out.writestr(name, data)

    log.info("Saved expanded checkpoint to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Expand MoE obs input dimension")
    parser.add_argument("--checkpoint", required=True, help="Path to model .zip")
    parser.add_argument("--output", required=True, help="Path for expanded model .zip")
    parser.add_argument("--old-obs-dim", type=int, default=28, help="Current obs dim")
    parser.add_argument("--new-obs-dim", type=int, default=33, help="Target obs dim")
    args = parser.parse_args()

    log.info("Loading checkpoint: %s", args.checkpoint)
    sd = load_policy_state_dict(args.checkpoint)

    for key, val in sd.items():
        if val.dim() == 2 and val.shape[1] == args.old_obs_dim:
            log.info("  Input layer: %s %s", key, val.shape)

    new_sd = expand_input_dim(sd, args.old_obs_dim, args.new_obs_dim)

    for key, val in new_sd.items():
        if val.dim() == 2 and val.shape[1] == args.new_obs_dim:
            log.info("  Expanded: %s %s", key, val.shape)

    save_expanded_checkpoint(args.checkpoint, args.output, new_sd)

    old_params = sum(v.numel() for v in sd.values())
    new_params = sum(v.numel() for v in new_sd.values())
    log.info("Params: %d -> %d (+%d, +%.1f%%)", old_params, new_params,
             new_params - old_params, (new_params - old_params) / old_params * 100)


if __name__ == "__main__":
    main()
