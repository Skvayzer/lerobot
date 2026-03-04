#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import train as train_main


@parser.wrap()
def train_policy_recap(cfg: TrainPipelineConfig):
    if cfg.policy is None:
        raise ValueError("Policy config is required.")

    cfg.policy.recap_enable = True
    labels_path = getattr(cfg.policy, "recap_labels_path", None)
    if labels_path is not None and str(labels_path).strip() != "":
        if not Path(labels_path).exists():
            raise FileNotFoundError(f"recap_labels_path does not exist: {labels_path}")

    train_main(cfg)


def main():
    train_policy_recap()


if __name__ == "__main__":
    main()

