#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.models.value_model import ValueModel, distributional_value_ce_loss
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.training.recap_utils import build_index_to_value, gather_values_by_index, load_labels_table
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train System-2 distributional value head.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to train_config.json")
    parser.add_argument("--labels_path", type=str, required=True, help="Labeled sidecar table (parquet/jsonl)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=0, help="0 -> use config batch_size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--target_bin_col", type=str, default="value_target_bin")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    init_logging()
    set_seed(args.seed)

    cfg = TrainPipelineConfig.from_pretrained(args.config_path, cli_args=[])
    cfg.validate()
    if cfg.policy is None:
        raise ValueError("Policy is missing in loaded config.")

    cfg.policy.recap_enable = True
    cfg.policy.recap_value_head_enable = True

    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs = {"dataset_stats": dataset.meta.stats}
    if cfg.policy.pretrained_path is not None:
        processor_kwargs = {
            "preprocessor_overrides": {
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            }
        }
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
    )

    labels_df = load_labels_table(args.labels_path)
    index_to_target_bin = build_index_to_value(labels_df, args.target_bin_col)
    if len(index_to_target_bin) == 0:
        raise ValueError("No labels were loaded from labels_path.")

    device = torch.device(cfg.policy.device if torch.cuda.is_available() else "cpu")
    policy.to(device)

    for param in policy.parameters():
        param.requires_grad = False
    value_head = getattr(getattr(policy, "_groot_model", None), "backbone", None)
    value_head = getattr(value_head, "value_head", None)
    if value_head is None:
        raise ValueError("Qwen value head is missing. Enable policy.recap_value_head_enable=true.")
    value_head.requires_grad_(True)

    model = ValueModel(policy).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg.batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not cfg.dataset.streaming,
        num_workers=int(args.num_workers),
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    iterator = iter(dataloader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "value_train_metrics.jsonl"

    policy.train()
    for step in range(1, int(args.steps) + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            raw_batch = next(iterator)

        batch = preprocessor(raw_batch)
        if "index" not in batch:
            raise ValueError("Preprocessed batch is missing 'index'.")

        target_bins = gather_values_by_index(
            batch["index"],
            index_to_value=index_to_target_bin,
            default_value=0.0,
            dtype=torch.float32,
            device=device,
        ).view(-1).round().long()

        logits, value_scalar = model(batch)
        loss = distributional_value_ce_loss(logits, target_bins)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % int(args.log_freq) == 0 or step == 1:
            metrics = {
                "step": step,
                "loss_ce": float(loss.detach().cpu().item()),
                "value_mean": float(value_scalar.detach().float().mean().cpu().item()),
                "target_bin_mean": float(target_bins.detach().float().mean().cpu().item()),
            }
            logging.info(metrics)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % int(args.save_freq) == 0 or step == int(args.steps):
            ckpt = {
                "step": step,
                "policy_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, output_dir / f"value_head_step_{step:06d}.pt")


if __name__ == "__main__":
    main()

