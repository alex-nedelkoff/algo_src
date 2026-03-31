"""Merge a pre-trained deceleration expert into the 5-expert MoE.

1. Expand existing experts from 28-dim to 56-dim input (zero-pad)
2. Expand router from 28-dim to 56-dim input
3. Expand value net from 28-dim to 56-dim input
4. Insert pre-trained decel expert as expert 5
5. Expand router output from 5→6

Usage:
    python -m control.merge_decel_expert \
        --moe-checkpoint outputs/2026-03-29/20-46-14/final_model.zip \
        --decel-checkpoint outputs/decel_expert/final_model.zip \
        --output outputs/merged_6expert/model.zip
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
    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        with zf.open("policy.pth") as f:
            return torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)


def expand_input_dim(weight: torch.Tensor, old_dim: int, new_dim: int) -> torch.Tensor:
    """Expand a linear layer's input dimension by zero-padding new columns."""
    if weight.shape[1] == new_dim:
        return weight.clone()
    assert weight.shape[1] == old_dim, f"Expected input dim {old_dim}, got {weight.shape[1]}"
    extra = torch.zeros(weight.shape[0], new_dim - old_dim, device=weight.device, dtype=weight.dtype)
    return torch.cat([weight.clone(), extra], dim=1)


def merge_state_dicts(
    moe_sd: dict,
    decel_sd: dict,
    old_obs_dim: int = 28,
    new_obs_dim: int = 56,
    old_n_experts: int = 5,
    noise_scale: float = 0.001,
) -> dict:
    """Merge MoE state dict with decel expert."""
    new_sd = {}

    for key, val in moe_sd.items():
        if key.startswith("experts."):
            # Expert first layer: expand input dim
            if ".net.0.weight" in key:
                new_sd[key] = expand_input_dim(val, old_obs_dim, new_obs_dim)
                log.info("Expanded %s: %s → %s", key, val.shape, new_sd[key].shape)
            else:
                new_sd[key] = val.clone()

        elif key == "router.net.0.weight":
            # Router first layer: expand input dim
            new_sd[key] = expand_input_dim(val, old_obs_dim, new_obs_dim)
            log.info("Expanded router input: %s → %s", val.shape, new_sd[key].shape)

        elif key == "router.net.2.weight":
            # Router output layer: expand from N→N+1 outputs
            # Add new row for expert N (decel expert)
            new_row = torch.zeros(1, val.shape[1], device=val.device, dtype=val.dtype)
            new_sd[key] = torch.cat([val.clone(), new_row], dim=0)
            log.info("Expanded router output: %s → %s", val.shape, new_sd[key].shape)

        elif key == "router.net.2.bias":
            # Router output bias: expand from N→N+1
            new_bias = torch.zeros(1, device=val.device, dtype=val.dtype)
            new_sd[key] = torch.cat([val.clone(), new_bias])
            log.info("Expanded router bias: %s → %s", val.shape, new_sd[key].shape)

        elif key == "value_net.0.weight":
            # Value net first layer: expand input dim
            new_sd[key] = expand_input_dim(val, old_obs_dim, new_obs_dim)
            log.info("Expanded value_net input: %s → %s", val.shape, new_sd[key].shape)

        else:
            new_sd[key] = val.clone()

    # Insert decel expert as expert N
    # The standalone MLP has keys like: mlp_extractor.policy_net.0.weight, action_net.weight, etc.
    # We need to map these to experts.{N}.net.{layer}.weight/bias
    expert_idx = old_n_experts
    layer_mapping = {
        "mlp_extractor.policy_net.0.weight": f"experts.{expert_idx}.net.0.weight",
        "mlp_extractor.policy_net.0.bias": f"experts.{expert_idx}.net.0.bias",
        "mlp_extractor.policy_net.2.weight": f"experts.{expert_idx}.net.2.weight",
        "mlp_extractor.policy_net.2.bias": f"experts.{expert_idx}.net.2.bias",
        "action_net.weight": f"experts.{expert_idx}.net.4.weight",
        "action_net.bias": f"experts.{expert_idx}.net.4.bias",
    }

    for decel_key, moe_key in layer_mapping.items():
        if decel_key in decel_sd:
            new_sd[moe_key] = decel_sd[decel_key].clone()
            log.info("Inserted decel %s → %s (%s)", decel_key, moe_key, new_sd[moe_key].shape)
        else:
            log.warning("Decel expert missing key: %s", decel_key)

    return new_sd


def save_checkpoint(original_path: str, output_path: str, new_sd: dict) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(original_path, "r") as zf_in:
        contents = {name: zf_in.read(name) for name in zf_in.namelist()}

    buf = io.BytesIO()
    torch.save(new_sd, buf)
    contents["policy.pth"] = buf.getvalue()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf_out:
        for name, data in contents.items():
            zf_out.writestr(name, data)

    log.info("Saved merged checkpoint to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge decel expert into MoE")
    parser.add_argument("--moe-checkpoint", required=True)
    parser.add_argument("--decel-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-obs-dim", type=int, default=28)
    parser.add_argument("--new-obs-dim", type=int, default=56)
    parser.add_argument("--old-n-experts", type=int, default=5)
    args = parser.parse_args()

    log.info("Loading MoE checkpoint: %s", args.moe_checkpoint)
    moe_sd = load_policy_state_dict(args.moe_checkpoint)

    log.info("Loading decel checkpoint: %s", args.decel_checkpoint)
    decel_sd = load_policy_state_dict(args.decel_checkpoint)

    # Log decel expert keys for debugging
    log.info("Decel expert keys: %s", sorted(decel_sd.keys()))

    new_sd = merge_state_dicts(
        moe_sd, decel_sd,
        old_obs_dim=args.old_obs_dim,
        new_obs_dim=args.new_obs_dim,
        old_n_experts=args.old_n_experts,
    )

    # Verify
    expert_keys = [k for k in new_sd if k.startswith("experts.")]
    expert_indices = set(int(k.split(".")[1]) for k in expert_keys)
    log.info("Merged checkpoint has %d experts: %s", len(expert_indices), sorted(expert_indices))

    old_params = sum(v.numel() for v in moe_sd.values())
    new_params = sum(v.numel() for v in new_sd.values())
    log.info("Params: %d → %d (+%d, +%.1f%%)", old_params, new_params,
             new_params - old_params, (new_params - old_params) / old_params * 100)

    save_checkpoint(args.moe_checkpoint, args.output, new_sd)


if __name__ == "__main__":
    main()
