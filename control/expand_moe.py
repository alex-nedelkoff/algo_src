"""Expand a 4-expert MoE checkpoint to 5 experts.

Clones the most-used expert, adds 1% noise for divergence,
expands the router output layer, and saves a new checkpoint.

Usage:
    python -m control.expand_moe \
        --checkpoint outputs/2026-03-29/00-24-04/final_model.zip \
        --output outputs/expanded_5expert/model.zip
"""
from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_policy_state_dict(checkpoint_path: str) -> dict:
    """Load the policy state dict from an SB3 checkpoint .zip."""
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        with zf.open("policy.pth") as f:
            return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)


def find_most_used_expert(state_dict: dict, n_experts: int = 4) -> int:
    """Estimate the most-used expert from router weights.

    Heuristic: the expert whose router output bias is highest is
    the one the router favors most. A full profiling run would be
    better but this is fast and good enough for cloning.
    """
    router_bias = state_dict.get("router.net.2.bias")
    if router_bias is not None:
        idx = int(torch.argmax(router_bias).item())
        log.info("Router output biases: %s", router_bias.detach().numpy())
        log.info("Most-favored expert by bias: %d", idx)
        return idx
    log.warning("No router bias found, defaulting to expert 0")
    return 0


def expand_state_dict(
    state_dict: dict,
    clone_expert_idx: int,
    noise_scale: float = 0.01,
    old_n_experts: int = 4,
) -> dict:
    """Expand a 4-expert state dict to 5 experts."""
    new_sd = {}

    for key, val in state_dict.items():
        # Clone expert weights
        if key.startswith(f"experts.{clone_expert_idx}."):
            # Copy original
            new_sd[key] = val.clone()
            # Create expert N (the clone)
            new_key = key.replace(f"experts.{clone_expert_idx}.", f"experts.{old_n_experts}.")
            new_sd[new_key] = val.clone() + noise_scale * torch.randn_like(val)
            log.info("Cloned %s → %s (%.1f%% noise)", key, new_key, noise_scale * 100)
        elif key.startswith("experts."):
            # Other experts: copy as-is
            new_sd[key] = val.clone()
        elif key == "router.net.2.weight":
            # Router final layer weight: [old_n_experts, hidden_dim] → [old_n_experts+1, hidden_dim]
            clone_row = val[clone_expert_idx].clone() + noise_scale * torch.randn_like(val[clone_expert_idx])
            new_sd[key] = torch.cat([val.clone(), clone_row.unsqueeze(0)], dim=0)
            log.info("Expanded router weight: %s → %s", val.shape, new_sd[key].shape)
        elif key == "router.net.2.bias":
            # Router final layer bias: [old_n_experts] → [old_n_experts+1]
            clone_bias = val[clone_expert_idx].clone() + noise_scale * torch.randn(1)
            new_sd[key] = torch.cat([val.clone(), clone_bias])
            log.info("Expanded router bias: %s → %s", val.shape, new_sd[key].shape)
        else:
            # Everything else (router hidden layers, value_net, log_std): copy as-is
            new_sd[key] = val.clone()

    return new_sd


def save_expanded_checkpoint(
    original_path: str,
    output_path: str,
    new_policy_sd: dict,
) -> None:
    """Save expanded checkpoint by replacing policy.pth in the zip."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Read original zip contents
    with zipfile.ZipFile(original_path, "r") as zf_in:
        names = zf_in.namelist()
        contents = {}
        for name in names:
            contents[name] = zf_in.read(name)

    # Replace policy.pth with expanded state dict
    buf = io.BytesIO()
    torch.save(new_policy_sd, buf)
    contents["policy.pth"] = buf.getvalue()

    # Write new zip
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
        for name, data in contents.items():
            zf_out.writestr(name, data)

    log.info("Saved expanded checkpoint to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Expand 4-expert MoE to 5 experts")
    parser.add_argument("--checkpoint", required=True, help="Path to 4-expert model .zip")
    parser.add_argument("--output", required=True, help="Path for expanded 5-expert model .zip")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Noise magnitude for clone divergence")
    parser.add_argument("--old-n-experts", type=int, default=4)
    args = parser.parse_args()

    log.info("Loading checkpoint: %s", args.checkpoint)
    sd = load_policy_state_dict(args.checkpoint)

    # Count current experts
    expert_keys = [k for k in sd if k.startswith("experts.")]
    expert_indices = set(int(k.split(".")[1]) for k in expert_keys)
    log.info("Found %d experts in checkpoint: %s", len(expert_indices), sorted(expert_indices))

    clone_idx = find_most_used_expert(sd, args.old_n_experts)
    log.info("Cloning expert %d with %.1f%% noise", clone_idx, args.noise_scale * 100)

    new_sd = expand_state_dict(sd, clone_idx, args.noise_scale, args.old_n_experts)

    # Verify new expert count
    new_expert_keys = [k for k in new_sd if k.startswith("experts.")]
    new_expert_indices = set(int(k.split(".")[1]) for k in new_expert_keys)
    log.info("New checkpoint has %d experts: %s", len(new_expert_indices), sorted(new_expert_indices))

    save_expanded_checkpoint(args.checkpoint, args.output, new_sd)

    # Summary
    old_params = sum(v.numel() for v in sd.values())
    new_params = sum(v.numel() for v in new_sd.values())
    log.info("Params: %d → %d (+%d, +%.1f%%)", old_params, new_params, new_params - old_params,
             (new_params - old_params) / old_params * 100)


if __name__ == "__main__":
    main()
