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
import dataclasses
import json
import logging
import math
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    prune_old_checkpoints,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


def _to_numpy_action(action: torch.Tensor) -> np.ndarray:
    if not isinstance(action, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor action, got {type(action)}")
    action = action.detach().float().cpu()
    if action.ndim == 0:
        action = action.view(1)
    if action.ndim > 1:
        action = action.reshape(-1, action.shape[-1])[0]
    return action.numpy()


def _plot_joint_trajectories(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting joint trajectories. "
            "Install it with `pip install matplotlib`."
        ) from exc

    n_steps, n_joints = gt.shape
    n_cols = 4
    n_rows = math.ceil(n_joints / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, max(12, n_rows * 2.8)), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    t = np.arange(n_steps)
    for joint_idx in range(n_joints):
        ax = axes[joint_idx]
        gt_vals = gt[:, joint_idx]
        pred_vals = pred[:, joint_idx]
        marker_size = 4.0 if n_steps > 1 else 7.0
        ax.plot(t, gt_vals, label="ground_truth", linewidth=1.4, marker="o", markersize=marker_size)
        ax.plot(
            t,
            pred_vals,
            label="predicted",
            linewidth=1.2,
            alpha=0.9,
            marker="x",
            markersize=marker_size,
        )
        if n_steps == 1:
            ax.set_xlim(-0.5, 0.5)
        ymin = float(min(np.min(gt_vals), np.min(pred_vals)))
        ymax = float(max(np.max(gt_vals), np.max(pred_vals)))
        if abs(ymax - ymin) < 1e-8:
            pad = max(1e-3, abs(ymin) * 0.05 + 1e-3)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(f"joint_{joint_idx}")
        ax.grid(alpha=0.25)
        if joint_idx % n_cols == 0:
            ax.set_ylabel("value")
        if joint_idx >= n_joints - n_cols:
            ax.set_xlabel("timestep")

    for ax in axes[n_joints:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_joint_error_mae(mae_per_joint: np.ndarray, out_path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting joint errors. Install it with `pip install matplotlib`."
        ) from exc

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(mae_per_joint.shape[0])
    ax.bar(x, mae_per_joint)
    ax.set_title(title)
    ax.set_xlabel("joint index")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    get_method = getattr(row, "get", None)
    if callable(get_method):
        try:
            return get_method(key, default)
        except Exception:
            pass
    try:
        return row[key]
    except Exception:
        return default


def _run_joint_reconstruction_eval(
    *,
    cfg: TrainPipelineConfig,
    dataset: Any,
    policy: PreTrainedPolicy,
    preprocessor: Any,
    postprocessor: Any,
    step: int,
    wandb_logger: WandBLogger | None,
) -> None:
    if cfg.joint_reconstruction_num_episodes <= 0:
        return

    episodes = getattr(getattr(dataset, "meta", None), "episodes", None)
    if episodes is None:
        logging.warning(
            "Skipping joint reconstruction eval: dataset metadata has no episodes table."
        )
        return

    total_episodes = len(episodes)
    if total_episodes == 0:
        logging.warning("Skipping joint reconstruction eval: dataset has no episodes.")
        return

    start_episode = min(cfg.joint_reconstruction_start_episode, total_episodes - 1)
    end_episode = min(start_episode + cfg.joint_reconstruction_num_episodes, total_episodes)
    if start_episode >= end_episode:
        logging.warning(
            "Skipping joint reconstruction eval: no episodes selected (start=%d, total=%d).",
            start_episode,
            total_episodes,
        )
        return

    step_id = get_step_identifier(step, cfg.steps)
    out_dir = cfg.output_dir / "eval" / "joint_reconstruction" / f"step_{step_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Running joint reconstruction eval on episodes [%d, %d) in mode=%s",
        start_episode,
        end_episode,
        cfg.joint_reconstruction_mode,
    )

    was_training = policy.training
    policy.eval()

    per_episode_results: list[dict[str, Any]] = []

    try:
        with torch.inference_mode():
            for episode_index in range(start_episode, end_episode):
                row = episodes[episode_index]
                start_idx = _row_get(row, "dataset_from_index")
                end_idx = _row_get(row, "dataset_to_index")

                try:
                    start_idx = int(start_idx)
                    end_idx = int(end_idx)
                except (TypeError, ValueError):
                    logging.warning(
                        "Skipping episode %d: invalid dataset range (%s, %s).",
                        episode_index,
                        start_idx,
                        end_idx,
                    )
                    continue

                if end_idx <= start_idx:
                    logging.warning(
                        "Skipping episode %d: empty dataset range [%d, %d).",
                        episode_index,
                        start_idx,
                        end_idx,
                    )
                    continue

                task_name = _row_get(row, "task_name", "unknown_task")
                if task_name is None:
                    tasks = _row_get(row, "tasks", [])
                    task_name = tasks[0] if isinstance(tasks, list) and tasks else "unknown_task"

                policy.reset()
                gt_actions: list[np.ndarray] = []
                pred_actions: list[np.ndarray] = []
                cot_traces: list[dict[str, Any]] = []

                for abs_idx in range(start_idx, end_idx):
                    sample = dataset[abs_idx]
                    if "action" not in sample:
                        continue

                    if cfg.joint_reconstruction_mode == "per_step":
                        policy.reset()

                    gt = _to_numpy_action(sample["action"])
                    observation = {k: v for k, v in sample.items() if k != "action"}
                    processed_obs = preprocessor(observation)
                    pred_action = policy.select_action(processed_obs)
                    pred_action = postprocessor(pred_action)
                    pred = _to_numpy_action(pred_action)

                    local_step_idx = abs_idx - start_idx
                    if (
                        cfg.joint_reconstruction_extract_cot
                        and (local_step_idx % cfg.joint_reconstruction_cot_every_n_steps == 0)
                        and has_method(policy, "extract_cot_trace")
                    ):
                        try:
                            traces = policy.extract_cot_trace(
                                processed_obs,
                                max_new_tokens=cfg.joint_reconstruction_cot_max_new_tokens,
                                do_sample=cfg.joint_reconstruction_cot_do_sample,
                                temperature=cfg.joint_reconstruction_cot_temperature,
                                top_p=cfg.joint_reconstruction_cot_top_p,
                            )
                            if traces:
                                trace = traces[0]
                                cot_traces.append(
                                    {
                                        "episode_index": int(episode_index),
                                        "dataset_index": int(abs_idx),
                                        "local_step_index": int(local_step_idx),
                                        "task_name": str(task_name),
                                        "raw_text": str(trace.get("raw_text", "")),
                                        "think_text": str(trace.get("think_text", "")),
                                        "summary_text": str(trace.get("summary_text", "")),
                                        "generated_tokens": int(trace.get("generated_tokens", 0)),
                                    }
                                )
                        except Exception as exc:
                            logging.warning(
                                "CoT extraction failed for episode %d step %d: %s",
                                episode_index,
                                local_step_idx,
                                exc,
                            )

                    gt_actions.append(gt)
                    pred_actions.append(pred)

                if not gt_actions or not pred_actions:
                    logging.warning(
                        "Skipping episode %d: no action samples collected.",
                        episode_index,
                    )
                    continue

                gt_arr = np.stack(gt_actions, axis=0)
                pred_arr = np.stack(pred_actions, axis=0)
                if gt_arr.shape != pred_arr.shape:
                    min_dim = min(gt_arr.shape[-1], pred_arr.shape[-1])
                    gt_arr = gt_arr[:, :min_dim]
                    pred_arr = pred_arr[:, :min_dim]
                    logging.warning(
                        "Episode %d dim mismatch; trimmed predicted/gt to %d dims.",
                        episode_index,
                        min_dim,
                    )

                mae_per_joint = np.mean(np.abs(pred_arr - gt_arr), axis=0)
                rmse_per_joint = np.sqrt(np.mean((pred_arr - gt_arr) ** 2, axis=0))

                ep_dir = out_dir / f"episode_{episode_index:04d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    ep_dir / "joint_trajectories.npz",
                    ground_truth=gt_arr,
                    predicted=pred_arr,
                )

                title = (
                    f"GT vs Pred Joint Trajectories | episode={episode_index} "
                    f"| task={task_name} | mode={cfg.joint_reconstruction_mode} "
                    f"| steps={gt_arr.shape[0]}"
                )
                trajectory_plot = ep_dir / "joint_trajectories.png"
                mae_plot = ep_dir / "joint_mae.png"
                _plot_joint_trajectories(gt_arr, pred_arr, trajectory_plot, title=title)
                _plot_joint_error_mae(
                    mae_per_joint,
                    mae_plot,
                    title=f"Per-joint MAE | episode={episode_index} | task={task_name}",
                )

                ep_metrics = {
                    "episode_index": int(episode_index),
                    "task_name": str(task_name),
                    "num_steps": int(gt_arr.shape[0]),
                    "num_joints": int(gt_arr.shape[1]),
                    "mae_mean": float(mae_per_joint.mean()),
                    "rmse_mean": float(rmse_per_joint.mean()),
                    "mae_per_joint": mae_per_joint.tolist(),
                    "rmse_per_joint": rmse_per_joint.tolist(),
                    "trajectory_plot": str(trajectory_plot),
                    "mae_plot": str(mae_plot),
                }
                with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                    json.dump(ep_metrics, f, indent=2)

                if cot_traces:
                    with open(ep_dir / "reasoning_traces.jsonl", "w", encoding="utf-8") as f:
                        for row in cot_traces:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ep_metrics["num_cot_traces"] = len(cot_traces)
                    with open(ep_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(ep_metrics, f, indent=2)

                per_episode_results.append(ep_metrics)
    finally:
        if was_training:
            policy.train()

    if not per_episode_results:
        logging.warning("Joint reconstruction eval finished with no valid episodes.")
        return

    mae_values = [m["mae_mean"] for m in per_episode_results]
    rmse_values = [m["rmse_mean"] for m in per_episode_results]
    summary = {
        "step": int(step),
        "mode": cfg.joint_reconstruction_mode,
        "num_episodes": len(per_episode_results),
        "start_episode": int(start_episode),
        "end_episode": int(end_episode),
        "mae_mean_avg": float(np.mean(mae_values)),
        "mae_mean_std": float(np.std(mae_values)),
        "rmse_mean_avg": float(np.mean(rmse_values)),
        "rmse_mean_std": float(np.std(rmse_values)),
        "episodes": per_episode_results,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        "Joint reconstruction eval complete: episodes=%d, mae_mean_avg=%.6f, rmse_mean_avg=%.6f, out_dir=%s",
        summary["num_episodes"],
        summary["mae_mean_avg"],
        summary["rmse_mean_avg"],
        out_dir,
    )

    if wandb_logger:
        wandb_logger.log_dict(
            {
                "joint_reconstruction/num_episodes": summary["num_episodes"],
                "joint_reconstruction/mae_mean_avg": summary["mae_mean_avg"],
                "joint_reconstruction/mae_mean_std": summary["mae_mean_std"],
                "joint_reconstruction/rmse_mean_avg": summary["rmse_mean_avg"],
                "joint_reconstruction/rmse_mean_std": summary["rmse_mean_std"],
            },
            step=step,
            mode="eval",
        )
        for episode_metrics in per_episode_results:
            ep_idx = episode_metrics["episode_index"]
            wandb_logger.log_image(
                episode_metrics["trajectory_plot"],
                step=step,
                mode="eval",
                key=f"joint_reconstruction/episode_{ep_idx:04d}_trajectory",
            )
            wandb_logger.log_image(
                episode_metrics["mae_plot"],
                step=step,
                mode="eval",
                key=f"joint_reconstruction/episode_{ep_idx:04d}_mae",
            )


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    last_joint_reconstruction_eval_step: int | None = None

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                removed_ckpts = prune_old_checkpoints(checkpoint_dir.parent, cfg.keep_last_n_checkpoints)
                if removed_ckpts:
                    logging.info(
                        "Pruned old checkpoints (kept last %d): %s",
                        cfg.keep_last_n_checkpoints,
                        [p.name for p in removed_ckpts],
                    )
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.joint_reconstruction_num_episodes > 0 and cfg.joint_reconstruction_eval_on_save and is_saving_step:
            if is_main_process:
                _run_joint_reconstruction_eval(
                    cfg=cfg,
                    dataset=dataset,
                    policy=accelerator.unwrap_model(policy),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    step=step,
                    wandb_logger=wandb_logger,
                )
                last_joint_reconstruction_eval_step = step
            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if cfg.joint_reconstruction_num_episodes > 0 and is_main_process and last_joint_reconstruction_eval_step != step:
        _run_joint_reconstruction_eval(
            cfg=cfg,
            dataset=dataset,
            policy=accelerator.unwrap_model(policy),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            step=step,
            wandb_logger=wandb_logger,
        )

    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
