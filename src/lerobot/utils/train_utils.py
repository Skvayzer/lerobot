#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import shutil
from pathlib import Path

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.utils import load_json, write_json
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.random_utils import load_rng_state, save_rng_state


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def prune_old_checkpoints(checkpoints_dir: Path, keep_last_n: int | None) -> list[Path]:
    """
    Delete old numbered checkpoint directories, keeping only the latest N.

    This never deletes the `last` symlink and only considers numeric checkpoint directories
    like `000500`, `010000`, etc.
    """
    if keep_last_n is None:
        return []

    ckpt_dirs = sorted(
        [
            p
            for p in checkpoints_dir.iterdir()
            if p.is_dir() and not p.is_symlink() and p.name.isdigit()
        ],
        key=lambda p: p.name,
    )
    if len(ckpt_dirs) <= keep_last_n:
        return []

    to_remove = ckpt_dirs[:-keep_last_n]
    for old_dir in to_remove:
        shutil.rmtree(old_dir)
    return to_remove


def _get_stat_dim(stat_value) -> int:
    """Get the first dimension of a stats value (numpy array, tensor, or list)."""
    import numpy as np
    import torch

    if isinstance(stat_value, (np.ndarray, torch.Tensor)):
        return stat_value.shape[0]
    return len(stat_value)


def validate_dataset_stats(dataset_stats: dict, dataset_info: dict) -> None:
    """Validate that dataset stats dimensions match the feature shapes in info.json.

    Raises ValueError if a mismatch is detected, preventing silent training on
    wrong normalization stats (e.g. stale meta/ from a different end-effector).
    """
    features = dataset_info.get("features", {})
    for key in ("action", "observation.state"):
        if key not in dataset_stats or key not in features:
            continue
        expected_shape = tuple(features[key].get("shape", ()))
        if not expected_shape:
            continue
        expected_dim = expected_shape[0]
        stat_entry = dataset_stats[key]
        for stat_name in ("min", "max", "mean", "std"):
            if stat_name not in stat_entry:
                continue
            stat_dim = _get_stat_dim(stat_entry[stat_name])
            if stat_dim != expected_dim:
                raise ValueError(
                    f"Stats/features dimension mismatch for '{key}.{stat_name}': "
                    f"stats dim={stat_dim}, info.json shape={expected_shape} "
                    f"(expected dim={expected_dim}). This likely means metadata was "
                    f"loaded from the wrong location (stale meta/ directory)."
                )


def reinject_dataset_stats(
    preprocessor: PolicyProcessorPipeline | None,
    postprocessor: PolicyProcessorPipeline | None,
    dataset_stats: dict | None,
) -> None:
    """Re-inject dataset stats into processor steps that use normalization.

    This is a defensive measure to ensure that the correct dataset statistics
    are always saved in checkpoints, even if something modifies them during
    training (e.g., code version mismatch after git pull on cluster).
    """
    if dataset_stats is None:
        return

    for pipeline in [preprocessor, postprocessor]:
        if pipeline is None:
            continue
        for step in pipeline.steps:
            if hasattr(step, "stats") and hasattr(step, "normalize_min_max"):
                # Before overwriting, check if existing stats have different dimensions
                # (indicates a potential dataset/model mismatch)
                if step.stats is not None:
                    old_action = step.stats.get("action", {}).get("min")
                    new_action = dataset_stats.get("action", {}).get("min")
                    if old_action is not None and new_action is not None:
                        old_dim = _get_stat_dim(old_action)
                        new_dim = _get_stat_dim(new_action)
                        if old_dim != new_dim:
                            logging.warning(
                                "reinject_dataset_stats: %s had action dim=%d, "
                                "overwriting with dim=%d from dataset. This dimension "
                                "change may indicate a dataset/end-effector mismatch.",
                                type(step).__name__, old_dim, new_dim,
                            )
                step.stats = dataset_stats
                new_action = dataset_stats.get("action", {}).get("min")
                if new_action is not None:
                    new_dim = _get_stat_dim(new_action)
                    logging.info(
                        "reinject_dataset_stats: set %s.stats action dim=%d (count=%s)",
                        type(step).__name__,
                        new_dim,
                        dataset_stats.get("action", {}).get("count"),
                    )


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        preprocessor: The preprocessor/pipeline to save. Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if cfg.peft is not None:
        # When using PEFT, policy.save_pretrained will only write the adapter weights + config, not the
        # policy config which we need for loading the model. In this case we'll write it ourselves.
        policy.config.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
