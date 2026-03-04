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
import inspect
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace

import datasets
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.humanoid_everyday_zip_dataset import HumanoidEverydayZipDataset
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def _canonical_robot_type(value: object) -> str:
    robot = str(value or "").strip().lower()
    compact = "".join(ch for ch in robot if ch.isalnum())
    if "g1" in compact:
        return "g1"
    if "h1" in compact:
        return "h1"
    return robot


def _normalize_task_name(name: str, case_sensitive: bool) -> str:
    text = str(name).strip()
    return text if case_sensitive else text.lower()


def _canonical_category_name(category: str) -> str:
    normalized = "".join(ch for ch in str(category).lower() if ch.isalnum())
    alias = {
        "articulateobject": "articulated",
        "basicmanipulation": "basic",
        "deformableobject": "deformable",
        "highprecision": "precision",
        "humanrobotinteraction": "hri",
        "locomanipulation": "locomanip",
        "tooluse": "tooluse",
    }
    return alias.get(normalized, normalized)


def _as_task_text(value: object, task_index_to_text: dict[int, str] | None = None) -> str:
    def _token_to_text(token: object) -> str:
        if isinstance(token, str):
            return token
        # Legacy episodes often store integer task_index values.
        if isinstance(token, int):
            if task_index_to_text is not None and token in task_index_to_text:
                return task_index_to_text[token]
            return str(token)
        if hasattr(token, "item"):
            with_value = token.item()
            if isinstance(with_value, int):
                if task_index_to_text is not None and with_value in task_index_to_text:
                    return task_index_to_text[with_value]
                return str(with_value)
        return str(token)

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return " ".join(_token_to_text(v) for v in value if v is not None)
    return _token_to_text(value)


def _resolve_filtered_episodes(cfg: TrainPipelineConfig, ds_meta: LeRobotDatasetMetadata) -> list[int] | None:
    exact_task_names = cfg.dataset.task_names or []
    task_categories = cfg.dataset.task_categories or []
    robot_types = cfg.dataset.robot_types or []
    include_keywords = cfg.dataset.task_include_keywords or []
    exclude_keywords = cfg.dataset.task_exclude_keywords or []
    if (
        not exact_task_names
        and not task_categories
        and not robot_types
        and not include_keywords
        and not exclude_keywords
    ):
        return cfg.dataset.episodes

    need_task_column = bool(exact_task_names or task_categories or include_keywords or exclude_keywords)
    if need_task_column and "tasks" not in ds_meta.episodes.features:
        raise ValueError(
            "Dataset episode metadata does not contain a 'tasks' column, but task keyword filtering was requested."
        )

    case_sensitive = cfg.dataset.task_keyword_case_sensitive
    task_name_norm = {_normalize_task_name(name, case_sensitive) for name in exact_task_names}
    category_norm = {_canonical_category_name(cat) for cat in task_categories}
    robot_types_norm = {_canonical_robot_type(rt) for rt in robot_types}
    include_norm = include_keywords if case_sensitive else [k.lower() for k in include_keywords]
    exclude_norm = exclude_keywords if case_sensitive else [k.lower() for k in exclude_keywords]
    task_index_to_text: dict[int, str] = {}
    task_index_to_category: dict[int, str] = {}
    try:
        dataset_robot_type = _canonical_robot_type(getattr(ds_meta, "robot_type", ""))
    except Exception:
        dataset_robot_type = ""
    if ds_meta.tasks is not None and "task_index" in ds_meta.tasks.columns:
        task_index_values = ds_meta.tasks["task_index"].tolist()
        for task_text, task_idx in zip(ds_meta.tasks.index.tolist(), task_index_values, strict=False):
            task_idx_int = int(task_idx)
            task_index_to_text[task_idx_int] = str(task_text)
            if "category" in ds_meta.tasks.columns:
                raw_category = ds_meta.tasks.at[task_text, "category"]
                task_index_to_category[task_idx_int] = str(raw_category)

    base_episodes = (
        cfg.dataset.episodes if cfg.dataset.episodes is not None else list(range(len(ds_meta.episodes)))
    )
    filtered_episodes: list[int] = []

    for ep_idx in base_episodes:
        episode = ds_meta.episodes[ep_idx]
        episode_tasks_raw = episode.get("tasks")
        if isinstance(episode_tasks_raw, (list, tuple, set)):
            task_tokens = list(episode_tasks_raw)
        elif episode_tasks_raw is None:
            task_tokens = []
        else:
            task_tokens = [episode_tasks_raw]

        episode_task_full_names: list[str] = []
        episode_task_bare_names: list[str] = []
        episode_task_categories: list[str] = []
        for token in task_tokens:
            full_name = ""
            category_name = ""
            if isinstance(token, int):
                full_name = task_index_to_text.get(token, str(token))
                category_name = task_index_to_category.get(token, "")
            elif hasattr(token, "item") and isinstance(token.item(), int):
                token_int = int(token.item())
                full_name = task_index_to_text.get(token_int, str(token_int))
                category_name = task_index_to_category.get(token_int, "")
            else:
                full_name = str(token)

            if full_name:
                episode_task_full_names.append(full_name)
                bare_name = full_name.split("/", 1)[1] if "/" in full_name else full_name
                episode_task_bare_names.append(bare_name)
                if not category_name and "/" in full_name:
                    category_name = full_name.split("/", 1)[0]
                if category_name:
                    episode_task_categories.append(category_name)

        if task_name_norm:
            episode_name_norm = {
                _normalize_task_name(name, case_sensitive)
                for name in [*episode_task_full_names, *episode_task_bare_names]
            }
            if not (task_name_norm & episode_name_norm):
                continue

        if category_norm:
            episode_category_norm = {_canonical_category_name(cat) for cat in episode_task_categories}
            if not (category_norm & episode_category_norm):
                continue

        if robot_types_norm:
            episode_robot_raw = episode.get("robot_type", "")
            episode_robot = _canonical_robot_type(episode_robot_raw)
            if not episode_robot and dataset_robot_type:
                episode_robot = dataset_robot_type
            if episode_robot not in robot_types_norm:
                continue

        episode_robot_text = _as_task_text(episode.get("robot_type"))
        if not episode_robot_text and dataset_robot_type:
            episode_robot_text = dataset_robot_type

        text_parts = [
            _as_task_text(episode_tasks_raw, task_index_to_text=task_index_to_text),
            episode_robot_text,
            _as_task_text(episode.get("instruction")),
        ]
        # Alias robot type to hand keyword used by Humanoid-Everyday users.
        robot_type = _canonical_robot_type(episode_robot_text)
        if robot_type == "g1":
            text_parts.extend(["dex3", "dex-3"])
        episode_text = " ".join(part for part in text_parts if part)
        haystack = episode_text if case_sensitive else episode_text.lower()

        include_ok = True if not include_norm else any(k in haystack for k in include_norm)
        exclude_hit = any(k in haystack for k in exclude_norm)
        if include_ok and not exclude_hit:
            filtered_episodes.append(ep_idx)

    if not filtered_episodes:
        raise ValueError(
            "Task keyword filtering matched 0 episodes. "
            f"include={include_keywords}, exclude={exclude_keywords}"
        )

    logging.info(
        "Applied task keyword filtering on episodes: %d -> %d "
        "(task_names=%s, task_categories=%s, robot_types=%s, include=%s, exclude=%s, case_sensitive=%s)",
        len(base_episodes),
        len(filtered_episodes),
        exact_task_names,
        task_categories,
        robot_types,
        include_keywords,
        exclude_keywords,
        case_sensitive,
    )
    return filtered_episodes


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _safe_scalar(value: object) -> object:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _build_multi_dataset_episodes(dataset: MultiLeRobotDataset) -> datasets.Dataset | None:
    rows = {
        "dataset_from_index": [],
        "dataset_to_index": [],
        "task_name": [],
        "tasks": [],
        "robot_type": [],
        "instruction": [],
        "dataset_index": [],
        "source_repo_id": [],
    }

    global_frame_offset = 0
    for dataset_index, sub_dataset in enumerate(dataset._datasets):
        episodes = getattr(getattr(sub_dataset, "meta", None), "episodes", None)
        if episodes is None:
            global_frame_offset += int(sub_dataset.num_frames)
            continue

        for episode_index in range(len(episodes)):
            episode_row = episodes[episode_index]
            start_idx = int(_safe_scalar(episode_row.get("dataset_from_index", 0)))
            end_idx = int(_safe_scalar(episode_row.get("dataset_to_index", 0)))
            rows["dataset_from_index"].append(start_idx + global_frame_offset)
            rows["dataset_to_index"].append(end_idx + global_frame_offset)
            rows["task_name"].append(_safe_scalar(episode_row.get("task_name")))
            rows["tasks"].append(_safe_scalar(episode_row.get("tasks")))
            rows["robot_type"].append(_safe_scalar(episode_row.get("robot_type")))
            rows["instruction"].append(_safe_scalar(episode_row.get("instruction")))
            rows["dataset_index"].append(dataset_index)
            rows["source_repo_id"].append(sub_dataset.repo_id)

        global_frame_offset += int(sub_dataset.num_frames)

    if len(rows["dataset_from_index"]) == 0:
        return None

    return datasets.Dataset.from_dict(rows)


def _build_multi_dataset_meta(dataset: MultiLeRobotDataset) -> SimpleNamespace:
    if len(dataset._datasets) == 0:
        raise ValueError("MultiLeRobotDataset has no sub-datasets.")

    first_meta = dataset._datasets[0].meta
    disabled_features = set(getattr(dataset, "disabled_features", set()))
    merged_features = {
        key: value
        for key, value in first_meta.features.items()
        if key not in disabled_features
    }
    camera_keys = [
        key
        for key, feature_spec in merged_features.items()
        if feature_spec.get("dtype") in {"image", "video"}
    ]

    return SimpleNamespace(
        repo_id=list(dataset.repo_ids),
        features=merged_features,
        stats=dataset.stats,
        camera_keys=camera_keys,
        fps=getattr(first_meta, "fps", None),
        robot_type=getattr(first_meta, "robot_type", None),
        episodes=_build_multi_dataset_episodes(dataset),
    )


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | MultiLeRobotDataset | StreamingLeRobotDataset | HumanoidEverydayZipDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )
    cfg.dataset.repo_id = cfg.dataset.resolve_repo_id()
    if cfg.dataset.dex3_dataset:
        logging.info(
            "Resolved dataset.dex3_dataset='%s' to repo_id='%s'",
            cfg.dataset.dex3_dataset,
            cfg.dataset.repo_id,
        )

    if cfg.dataset.zip_paths or cfg.dataset.zip_glob:
        if cfg.dataset.streaming:
            raise ValueError("`dataset.streaming=true` is not supported with zip dataset loading.")

        dataset = HumanoidEverydayZipDataset(
            repo_id=cfg.dataset.repo_id,
            zip_paths=cfg.dataset.zip_paths,
            zip_glob=cfg.dataset.zip_glob,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            image_transforms=image_transforms,
            task_names=cfg.dataset.task_names,
            task_categories=cfg.dataset.task_categories,
            robot_types=cfg.dataset.robot_types,
            task_include_keywords=cfg.dataset.task_include_keywords,
            task_exclude_keywords=cfg.dataset.task_exclude_keywords,
            task_keyword_case_sensitive=cfg.dataset.task_keyword_case_sensitive,
            tolerance_s=cfg.tolerance_s,
        )
        delta_timestamps = resolve_delta_timestamps(cfg.policy, dataset.meta)
        dataset.set_delta_timestamps(delta_timestamps, tolerance_s=cfg.tolerance_s)
    elif isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )
        resolved_episodes = _resolve_filtered_episodes(cfg, ds_meta)
        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=resolved_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.dataset.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=resolved_episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.dataset.tolerance_s,
            )
    else:
        if cfg.dataset.streaming:
            raise ValueError("`dataset.streaming=true` is not supported with multi-dataset loading.")

        repo_ids = [str(repo_id).strip() for repo_id in cfg.dataset.repo_id if str(repo_id).strip()]
        if len(repo_ids) == 0:
            raise ValueError("No valid dataset repo_ids were provided for multi-dataset loading.")

        if cfg.dataset.revision is not None:
            logging.warning(
                "dataset.revision is ignored in multi-dataset mode because MultiLeRobotDataset "
                "currently loads each sub-dataset using default revision semantics."
            )

        episodes_per_repo = {}
        tolerances_s = {}
        delta_timestamps = None
        base_root = Path(cfg.dataset.root) if cfg.dataset.root is not None else None
        reference_fps = None

        for repo_id in repo_ids:
            meta_root = str(base_root / repo_id) if base_root is not None else None
            ds_meta = LeRobotDatasetMetadata(
                repo_id,
                root=meta_root,
                revision=cfg.dataset.revision,
            )
            if reference_fps is None:
                reference_fps = ds_meta.fps
                delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
            elif ds_meta.fps != reference_fps:
                raise ValueError(
                    "All datasets in multi-dataset mode must have the same fps for consistent delta_timestamps. "
                    f"Expected {reference_fps}, got {ds_meta.fps} for repo '{repo_id}'."
                )

            episodes_per_repo[repo_id] = _resolve_filtered_episodes(cfg, ds_meta)
            tolerances_s[repo_id] = cfg.tolerance_s

        multi_dataset_kwargs = {
            "root": cfg.dataset.root,
            "episodes": episodes_per_repo,
            "image_transforms": image_transforms,
            "delta_timestamps": delta_timestamps,
            "video_backend": cfg.dataset.video_backend,
        }
        multi_dataset_sig = inspect.signature(MultiLeRobotDataset.__init__).parameters
        if "tolerances_s" in multi_dataset_sig:
            multi_dataset_kwargs["tolerances_s"] = tolerances_s
        if "tolerance_s" in multi_dataset_sig:
            multi_dataset_kwargs["tolerance_s"] = cfg.dataset.tolerance_s

        dataset = MultiLeRobotDataset(repo_ids, **multi_dataset_kwargs)
        dataset.meta = _build_multi_dataset_meta(dataset)
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            dataset.meta.stats.setdefault(key, {})
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset
