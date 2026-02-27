#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.humanoid_everyday_zip_dataset import HumanoidEverydayZipDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline trajectory reconstruction on a full episode and plot "
            "ground-truth vs predicted joint trajectories."
        )
    )
    parser.add_argument(
        "--policy.path",
        dest="policy_path",
        required=True,
        help="Path to checkpoint pretrained_model directory.",
    )
    parser.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    parser.add_argument("--dataset.zip_glob", dest="dataset_zip_glob", required=True)
    parser.add_argument(
        "--dataset.robot_types",
        dest="dataset_robot_types",
        default='["g1"]',
        help='JSON list, e.g. \'["g1"]\'.',
    )
    parser.add_argument("--tolerance_s", type=float, default=1e-2)
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index in filtered dataset.")
    parser.add_argument(
        "--mode",
        choices=("per_step", "queue_rollout"),
        default="per_step",
        help=(
            "per_step: recompute action each timestep (policy reset every step). "
            "queue_rollout: use policy action queue behavior."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where plots/metrics are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cpu", "cuda"),
        help="Optional override for policy device.",
    )
    parser.add_argument(
        "--extract_cot",
        action="store_true",
        help="If set, generate Qwen reasoning traces for selected timesteps.",
    )
    parser.add_argument("--cot_max_new_tokens", type=int, default=64)
    parser.add_argument("--cot_every_n_steps", type=int, default=1)
    parser.add_argument("--cot_do_sample", action="store_true")
    parser.add_argument("--cot_temperature", type=float, default=0.7)
    parser.add_argument("--cot_top_p", type=float, default=0.9)
    return parser.parse_args()


def _parse_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse list from: {value}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got: {type(parsed)}")
    return [str(x) for x in parsed]


def _resolve_episode_meta(dataset: HumanoidEverydayZipDataset, episode_index: int) -> dict[str, Any]:
    episodes = dataset.meta.episodes
    total = len(episodes)
    if episode_index < 0 or episode_index >= total:
        raise IndexError(f"episode_index={episode_index} out of range [0, {total - 1}]")

    row = episodes[episode_index]

    task_name = row.get("task_name")
    if task_name is None:
        tasks = row.get("tasks")
        if isinstance(tasks, list) and tasks:
            task_name = tasks[0]
        else:
            task_name = "unknown_task"

    task_full_name = row.get("task_full_name", task_name)
    category_name = row.get("category_name", "unknown_category")

    return {
        "episode_index": int(row["episode_index"]),
        "task_name": str(task_name),
        "task_full_name": str(task_full_name),
        "category_name": str(category_name),
        "robot_type": str(row["robot_type"]),
        "dataset_from_index": int(row["dataset_from_index"]),
        "dataset_to_index": int(row["dataset_to_index"]),
        "length": int(row["length"]),
    }


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
            "matplotlib is required for plotting. Install it in your env: "
            "`pip install matplotlib`"
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
        # Always draw markers so 1-step trajectories are visible.
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
            # Give a little x-range so the single marker does not collapse at center.
            ax.set_xlim(-0.5, 0.5)

        # Add a small y-padding for near-constant/identical values.
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
            "matplotlib is required for plotting. Install it in your env: "
            "`pip install matplotlib`"
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")

    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")

    robot_types = _parse_json_list(args.dataset_robot_types)
    dataset = HumanoidEverydayZipDataset(
        repo_id=args.dataset_repo_id,
        zip_glob=args.dataset_zip_glob,
        robot_types=robot_types,
        tolerance_s=args.tolerance_s,
    )
    ep_meta = _resolve_episode_meta(dataset, args.episode_index)
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]
    total_steps = end_idx - start_idx
    if total_steps <= 1:
        logging.warning(
            "Episode has only %d timestep. Plot will contain one GT point and one predicted point per joint.",
            total_steps,
        )

    cfg = PreTrainedConfig.from_pretrained(str(policy_path))
    cfg.pretrained_path = policy_path

    if args.device is not None:
        cfg.device = args.device
    elif cfg.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA is not available. Falling back to CPU.")
        cfg.device = "cpu"

    if cfg.device != "cuda" and getattr(cfg, "attn_implementation", None) == "flash_attention_2":
        logging.warning(
            "flash_attention_2 requires CUDA. Switching attn_implementation to 'eager' for %s device.",
            cfg.device,
        )
        cfg.attn_implementation = "eager"

    if cfg.device == "cpu" and getattr(cfg, "use_bf16", False):
        logging.warning("BF16 on CPU can cause dtype issues. Disabling use_bf16/use_amp for CPU evaluation.")
        cfg.use_bf16 = False
        cfg.use_amp = False

    policy = make_policy(cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    policy.eval()
    policy.reset()

    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    cot_traces: list[dict[str, Any]] = []

    logging.info(
        "Running reconstruction on episode=%d task='%s' robot='%s' steps=%d mode=%s",
        args.episode_index,
        ep_meta["task_name"],
        ep_meta["robot_type"],
        total_steps,
        args.mode,
    )

    with torch.inference_mode():
        for local_step_idx, abs_idx in enumerate(range(start_idx, end_idx)):
            sample = dataset[abs_idx]
            gt = _to_numpy_action(sample["action"])

            # Remove ground-truth action to avoid leakage in inference path.
            observation = {k: v for k, v in sample.items() if k != "action"}
            if args.mode == "per_step":
                policy.reset()

            processed_obs = preprocessor(observation)
            pred_action = policy.select_action(processed_obs)
            pred_action = postprocessor(pred_action)
            pred = _to_numpy_action(pred_action)

            if (
                args.extract_cot
                and local_step_idx % max(1, args.cot_every_n_steps) == 0
                and hasattr(policy, "extract_cot_trace")
            ):
                try:
                    traces = policy.extract_cot_trace(
                        processed_obs,
                        max_new_tokens=args.cot_max_new_tokens,
                        do_sample=args.cot_do_sample,
                        temperature=args.cot_temperature,
                        top_p=args.cot_top_p,
                    )
                    if traces:
                        trace = traces[0]
                        cot_traces.append(
                            {
                                "episode_index": int(args.episode_index),
                                "dataset_index": int(abs_idx),
                                "local_step_index": int(local_step_idx),
                                "raw_text": str(trace.get("raw_text", "")),
                                "think_text": str(trace.get("think_text", "")),
                                "summary_text": str(trace.get("summary_text", "")),
                                "generated_tokens": int(trace.get("generated_tokens", 0)),
                            }
                        )
                except Exception as exc:
                    logging.warning("Failed CoT extraction at step %d: %s", local_step_idx, exc)

            gt_actions.append(gt)
            pred_actions.append(pred)

    gt_arr = np.stack(gt_actions, axis=0)
    pred_arr = np.stack(pred_actions, axis=0)
    if gt_arr.shape != pred_arr.shape:
        min_dim = min(gt_arr.shape[-1], pred_arr.shape[-1])
        gt_arr = gt_arr[:, :min_dim]
        pred_arr = pred_arr[:, :min_dim]
        logging.warning("Pred/GT dim mismatch detected; trimmed to %d dims.", min_dim)

    mae_per_joint = np.mean(np.abs(pred_arr - gt_arr), axis=0)
    rmse_per_joint = np.sqrt(np.mean((pred_arr - gt_arr) ** 2, axis=0))
    metrics = {
        "episode_index": int(args.episode_index),
        "task_name": ep_meta["task_name"],
        "robot_type": ep_meta["robot_type"],
        "num_steps": int(gt_arr.shape[0]),
        "num_joints": int(gt_arr.shape[1]),
        "mode": args.mode,
        "mae_mean": float(mae_per_joint.mean()),
        "rmse_mean": float(rmse_per_joint.mean()),
        "mae_per_joint": mae_per_joint.tolist(),
        "rmse_per_joint": rmse_per_joint.tolist(),
        "note": (
            "Comparison is between predicted policy action and ground-truth action "
            "(joint targets) from the dataset."
        ),
    }

    default_out = (
        Path("outputs")
        / "eval"
        / "joint_reconstruction"
        / f"episode_{args.episode_index:04d}_{ep_meta['task_name']}"
    )
    out_dir = (args.output_dir or default_out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "joint_trajectories.npz", ground_truth=gt_arr, predicted=pred_arr)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if cot_traces:
        with open(out_dir / "reasoning_traces.jsonl", "w", encoding="utf-8") as f:
            for row in cot_traces:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics["num_cot_traces"] = len(cot_traces)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    title = (
        f"GT vs Predicted Joint Trajectories | episode={args.episode_index} "
        f"| task={ep_meta['task_name']} | mode={args.mode} | steps={gt_arr.shape[0]}"
    )
    _plot_joint_trajectories(gt_arr, pred_arr, out_dir / "joint_trajectories.png", title=title)
    _plot_joint_error_mae(
        mae_per_joint,
        out_dir / "joint_mae.png",
        title=f"Per-joint MAE | episode={args.episode_index} | task={ep_meta['task_name']}",
    )

    logging.info("Saved outputs to: %s", out_dir.resolve())
    logging.info("MAE(mean)=%.6f RMSE(mean)=%.6f", metrics["mae_mean"], metrics["rmse_mean"])


if __name__ == "__main__":
    main()
