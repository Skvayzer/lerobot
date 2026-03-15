#!/usr/bin/env python3
"""Compute min/max normalization stats for relative (state-delta) actions.

Usage:
    python scripts/compute_relative_action_stats.py \
        --dataset unitreerobotics/G1_Dex3_BlockStacking_Dataset \
        --output relative_action_stats_dex3_blockstack.json

The output JSON can be passed to CraftNet training via:
    --policy.relative_action_stats_path=relative_action_stats_dex3_blockstack.json
"""
import argparse
import json
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HF dataset repo ID")
    parser.add_argument("--root", default=None, help="Local dataset root (optional)")
    parser.add_argument("--output", default="relative_action_stats.json")
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Max samples to process (0=all)")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    load_kwargs = {"repo_id": args.dataset}
    if args.root:
        load_kwargs["root"] = args.root
    ds = LeRobotDataset(**load_kwargs)
    n = len(ds) if args.max_samples <= 0 else min(len(ds), args.max_samples)
    print(f"Dataset: {args.dataset} ({len(ds)} samples, processing {n})")

    all_relative = []
    for i in range(n):
        if i % 10000 == 0 and i > 0:
            print(f"  processed {i}/{n}...")
        sample = ds[i]
        state = sample["observation.state"]
        action = sample["action"]

        if state.dim() == 2:
            state = state[-1]
        if action.dim() == 1:
            action = action.unsqueeze(0)

        d = min(state.shape[-1], action.shape[-1])
        relative = action[..., :d] - state[:d].unsqueeze(0)
        all_relative.append(relative)

    all_relative = torch.cat(all_relative, dim=0)
    print(f"\nRelative action tensor: {all_relative.shape}")
    print(f"  Mean:  {all_relative.mean(dim=0)[:6].tolist()}")
    print(f"  Std:   {all_relative.std(dim=0)[:6].tolist()}")
    print(f"  Min:   {all_relative.min(dim=0).values[:6].tolist()}")
    print(f"  Max:   {all_relative.max(dim=0).values[:6].tolist()}")
    print(f"  Range: {(all_relative.max(dim=0).values - all_relative.min(dim=0).values)[:6].tolist()}")

    mean_abs = all_relative.mean(dim=0).abs().mean().item()
    if mean_abs > 0.1:
        print(f"  WARNING: mean |value| {mean_abs:.4f} is high — expected near 0")
    else:
        print(f"  OK: mean near zero ({mean_abs:.6f})")

    stats = {
        "action": {
            "min": all_relative.min(dim=0).values.tolist(),
            "max": all_relative.max(dim=0).values.tolist(),
            "mean": all_relative.mean(dim=0).tolist(),
            "std": all_relative.std(dim=0).tolist(),
        }
    }
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
