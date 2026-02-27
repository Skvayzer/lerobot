#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

import bisect
import glob
import io
import json
import lzma
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from lerobot.datasets.transforms import ImageTransforms
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices


CAMERA_FIELD_CANDIDATES: dict[str, list[str]] = {
    "observation.images.head_rgb": [
        "head_rgb",
        "head_image",
        "head_camera_rgb",
        "image",
    ],
    "observation.images.left_wrist_rgb": [
        "left_wrist_rgb",
        "left_wrist_image",
        "left_hand_image",
        "left_image",
        "wrist_left_rgb",
    ],
    "observation.images.right_wrist_rgb": [
        "right_wrist_rgb",
        "right_wrist_image",
        "right_hand_image",
        "right_image",
        "wrist_right_rgb",
    ],
    "observation.images.head_depth": [
        "head_depth",
        "head_depth_image",
        "depth",
    ],
    "observation.images.left_wrist_depth": [
        "left_wrist_depth",
        "left_wrist_depth_image",
        "left_depth",
        "wrist_left_depth",
    ],
    "observation.images.right_wrist_depth": [
        "right_wrist_depth",
        "right_wrist_depth_image",
        "right_depth",
        "wrist_right_depth",
    ],
}

WRIST_CAMERA_KEYS = {
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
    "observation.images.left_wrist_depth",
    "observation.images.right_wrist_depth",
}

DEFAULT_CAMERA_PRIORITY = [
    "observation.images.head_rgb",
    "observation.images.left_wrist_rgb",
    "observation.images.right_wrist_rgb",
    "observation.images.head_depth",
    "observation.images.left_wrist_depth",
    "observation.images.right_wrist_depth",
]


def _normalize_text(text: str, case_sensitive: bool) -> str:
    value = str(text).strip()
    return value if case_sensitive else value.lower()


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


def _to_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for x in value:
            try:
                out.append(float(x))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _flatten_numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, dict):
        out: list[float] = []
        for _, inner in value.items():
            out.extend(_flatten_numeric_values(inner))
        return out

    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for inner in value:
            out.extend(_flatten_numeric_values(inner))
        return out

    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _extract_camera_paths(frame: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for obs_key, candidates in CAMERA_FIELD_CANDIDATES.items():
        for src_key in candidates:
            value = frame.get(src_key)
            if isinstance(value, str) and value.strip():
                resolved[obs_key] = value.strip()
                break
    return resolved


def _extract_extra_observation_vectors(frame: dict[str, Any]) -> dict[str, list[float]]:
    states = frame.get("states", {}) if isinstance(frame.get("states"), dict) else {}
    vectors: dict[str, list[float]] = {}

    leg_state = _to_float_list(states.get("leg_state"))
    if leg_state:
        vectors["observation.extra.leg_state"] = leg_state

    hand_pressure = _flatten_numeric_values(states.get("hand_pressure_state"))
    if hand_pressure:
        vectors["observation.extra.hand_pressure"] = hand_pressure

    imu = _flatten_numeric_values(states.get("imu"))
    if imu:
        vectors["observation.extra.imu"] = imu

    odometry = _flatten_numeric_values(states.get("odometry"))
    if odometry:
        vectors["observation.extra.odometry"] = odometry

    return vectors


def _infer_robot_type(frame: dict[str, Any]) -> str:
    robot_type = str(frame.get("robot_type", "")).strip().lower()
    if robot_type:
        return robot_type

    states = frame.get("states", {}) if isinstance(frame.get("states"), dict) else {}
    hand_state = _to_float_list(states.get("hand_state"))
    if len(hand_state) == 14:
        return "g1"
    if len(hand_state) == 12:
        return "h1"
    return "unknown"


def _extract_state_vector(frame: dict[str, Any]) -> list[float]:
    states = frame.get("states", {}) if isinstance(frame.get("states"), dict) else {}
    arm_state = _to_float_list(states.get("arm_state"))
    hand_state = _to_float_list(states.get("hand_state"))
    return arm_state + hand_state


def _extract_h1_hand_action(left_qpos: list[float], right_qpos: list[float]) -> list[float]:
    # H1 hand actions in Humanoid-Everyday require post-processing from internal representation.
    required_idx = [0, 2, 4, 6, 8, 9]
    if any(i >= len(left_qpos) for i in required_idx) or any(i >= len(right_qpos) for i in required_idx):
        return left_qpos + right_qpos

    right_hand_angles = [1.7 - right_qpos[i] for i in [4, 6, 2, 0]]
    right_hand_angles.append(1.2 - right_qpos[8])
    right_hand_angles.append(0.5 - right_qpos[9])
    left_hand_angles = [1.7 - left_qpos[i] for i in [4, 6, 2, 0]]
    left_hand_angles.append(1.2 - left_qpos[8])
    left_hand_angles.append(0.5 - left_qpos[9])
    return left_hand_angles + right_hand_angles


def _extract_action_vector(frame: dict[str, Any], robot_type: str) -> list[float]:
    actions = frame.get("actions", {}) if isinstance(frame.get("actions"), dict) else {}
    arm_actions = _to_float_list(actions.get("sol_q"))
    left_angles = _to_float_list(actions.get("left_angles"))
    right_angles = _to_float_list(actions.get("right_angles"))

    if robot_type.startswith("h1"):
        hand_actions = _extract_h1_hand_action(left_angles, right_angles)
    else:
        hand_actions = left_angles + right_angles

    return arm_actions + hand_actions


@dataclass
class _EpisodeDescriptor:
    json_path: Path
    episode_dir: Path
    task_name: str
    task_full_name: str
    category_name: str
    robot_type: str
    length: int


@dataclass
class _EpisodePayload:
    episode_index: int
    episode_dir: Path
    task_name: str
    robot_type: str
    dataset_from_index: int
    dataset_to_index: int
    camera_paths: dict[str, list[str]]
    states: np.ndarray
    extra_observations: dict[str, np.ndarray]
    actions: np.ndarray
    timestamps: np.ndarray


@dataclass
class HumanoidEverydayZipMeta:
    repo_id: str
    root: Path | None
    fps: int
    features: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, torch.Tensor]]
    episodes: datasets.Dataset
    tasks: pd.DataFrame
    robot_type: str = "mixed"

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def total_frames(self) -> int:
        return int(self.episodes["dataset_to_index"][-1]) if len(self.episodes) > 0 else 0

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)


class HumanoidEverydayZipDataset(Dataset):
    """Dataset adapter that reads Humanoid-Everyday task zip files directly.

    This adapter exposes the same runtime contract expected by `lerobot_train.py`:
    - `__getitem__` returns LeRobot-style keys (`observation.images.*`, `observation.state`, `action`, `task`, ...)
    - `meta` provides `features`, `stats`, `episodes`, `tasks`, and `camera_keys`
    """

    def __init__(
        self,
        repo_id: str,
        zip_paths: list[str] | None = None,
        zip_glob: str | None = None,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: ImageTransforms | None = None,
        task_names: list[str] | None = None,
        task_categories: list[str] | None = None,
        robot_types: list[str] | None = None,
        task_include_keywords: list[str] | None = None,
        task_exclude_keywords: list[str] | None = None,
        task_keyword_case_sensitive: bool = False,
        fps: int = 30,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
    ):
        super().__init__()

        self.repo_id = repo_id
        self.root = Path(root).expanduser() if root else None
        self.image_transforms = image_transforms
        self.fps = fps
        self.tolerance_s = tolerance_s
        self._to_tensor = transforms.ToTensor()

        self.delta_timestamps = delta_timestamps
        self.delta_indices: dict[str, list[int]] | None = None
        self._unsupported_delta_keys: set[str] = set()

        resolved_zip_paths = self._resolve_zip_paths(zip_paths=zip_paths, zip_glob=zip_glob)
        descriptors = self._discover_episode_descriptors(
            resolved_zip_paths=resolved_zip_paths,
            task_names=task_names,
            task_categories=task_categories,
            robot_types=robot_types,
            task_include_keywords=task_include_keywords,
            task_exclude_keywords=task_exclude_keywords,
            task_keyword_case_sensitive=task_keyword_case_sensitive,
        )

        if episodes is not None:
            valid = [idx for idx in episodes if 0 <= idx < len(descriptors)]
            descriptors = [descriptors[idx] for idx in valid]
            logging.info(
                "Applied explicit episode-index filter for zip dataset: requested=%d -> kept=%d",
                len(episodes),
                len(descriptors),
            )

        if len(descriptors) == 0:
            raise ValueError(
                "No episodes matched the provided zip dataset filters. "
                "Check zip paths and task/robot filters."
            )

        (
            self._episodes_payload,
            self._dataset_from_indices,
            self._dataset_to_indices,
            state_dim,
            action_dim,
            image_shape_hw,
            camera_keys,
            extra_observation_dims,
            tasks_df,
            episodes_ds,
            stats,
        ) = self._materialize_payload(descriptors)

        self._num_frames = self._dataset_to_indices[-1]
        self._image_shape_hw = image_shape_hw
        self._camera_keys = camera_keys
        self._extra_observation_dims = extra_observation_dims
        self._missing_modalities_warned: set[str] = set()
        self.episodes = list(range(len(self._episodes_payload)))
        self.meta = HumanoidEverydayZipMeta(
            repo_id=repo_id,
            root=self.root,
            fps=fps,
            features=self._build_features(
                state_dim=state_dim,
                action_dim=action_dim,
                image_shape_hw=image_shape_hw,
                camera_keys=camera_keys,
                extra_observation_dims=extra_observation_dims,
            ),
            stats=stats,
            episodes=episodes_ds,
            tasks=tasks_df,
            robot_type=self._infer_global_robot_type(descriptors),
        )

        self.set_delta_timestamps(self.delta_timestamps, tolerance_s=self.tolerance_s)

        logging.info(
            "Loaded Humanoid-Everyday zip dataset: zips=%d, episodes=%d, frames=%d, "
            "state_dim=%d, action_dim=%d",
            len(resolved_zip_paths),
            self.num_episodes,
            self.num_frames,
            state_dim,
            action_dim,
        )

    def set_delta_timestamps(
        self, delta_timestamps: dict[str, list[float]] | None, tolerance_s: float | None = None
    ) -> None:
        self.delta_timestamps = delta_timestamps
        self.delta_indices = None
        if self.delta_timestamps is None:
            return

        if tolerance_s is not None:
            self.tolerance_s = tolerance_s
        check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
        self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    @staticmethod
    def _resolve_zip_paths(zip_paths: list[str] | None, zip_glob: str | None) -> list[Path]:
        resolved: list[Path] = []

        for p in zip_paths or []:
            resolved.append(Path(p).expanduser().resolve())

        if zip_glob:
            pattern = str(Path(zip_glob).expanduser())
            resolved.extend(Path(p).resolve() for p in glob.glob(pattern, recursive=True))

        unique = sorted({path for path in resolved if path.suffix.lower() == ".zip"})
        if len(unique) == 0:
            raise ValueError(
                "No .zip files were provided. Set `dataset.zip_paths` or `dataset.zip_glob` "
                "to Humanoid-Everyday task zip files."
            )
        missing = [str(p) for p in unique if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Some zip files do not exist: {missing[:5]}")
        return unique

    def _extract_zip_if_needed(self, zip_path: Path) -> Path:
        if self.root is None:
            extract_dir = zip_path.with_suffix("")
        else:
            extract_dir = self.root / zip_path.stem

        has_json = extract_dir.exists() and any(extract_dir.rglob("data.json"))
        if has_json:
            return extract_dir

        extract_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Extracting zip dataset: %s -> %s", zip_path, extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = [m for m in zip_ref.namelist() if m and not m.startswith("__MACOSX")]
            zip_ref.extractall(extract_dir, members=members)
        return extract_dir

    @staticmethod
    def _derive_task_from_episode_dir(episode_dir: Path, extract_root: Path, zip_path: Path) -> tuple[str, str, str]:
        try:
            rel = episode_dir.relative_to(extract_root)
            parts = list(rel.parts)
        except ValueError:
            parts = list(episode_dir.parts)

        if parts and parts[-1].startswith("episode_"):
            parts = parts[:-1]

        if len(parts) == 0:
            task_name = zip_path.stem
            category_name = ""
        elif len(parts) == 1:
            task_name = parts[-1]
            category_name = ""
        else:
            task_name = parts[-1]
            category_name = parts[-2]

        task_full_name = f"{category_name}/{task_name}" if category_name else task_name
        return task_name, task_full_name, category_name

    @staticmethod
    def _episode_matches_filters(
        descriptor: _EpisodeDescriptor,
        task_names: list[str] | None,
        task_categories: list[str] | None,
        robot_types: list[str] | None,
        task_include_keywords: list[str] | None,
        task_exclude_keywords: list[str] | None,
        case_sensitive: bool,
    ) -> bool:
        task_names = task_names or []
        task_categories = task_categories or []
        robot_types = robot_types or []
        task_include_keywords = task_include_keywords or []
        task_exclude_keywords = task_exclude_keywords or []

        if task_names:
            wanted = {_normalize_text(name, case_sensitive) for name in task_names}
            candidates = {
                _normalize_text(descriptor.task_name, case_sensitive),
                _normalize_text(descriptor.task_full_name, case_sensitive),
            }
            if len(wanted & candidates) == 0:
                return False

        if task_categories:
            wanted = {_canonical_category_name(cat) for cat in task_categories}
            category = _canonical_category_name(descriptor.category_name)
            if category not in wanted:
                return False

        if robot_types:
            wanted = {str(rt).strip().lower() for rt in robot_types}
            if descriptor.robot_type.lower() not in wanted:
                return False

        haystack = " ".join(
            [
                descriptor.task_name,
                descriptor.task_full_name,
                descriptor.category_name,
                descriptor.robot_type,
            ]
        )
        haystack = haystack if case_sensitive else haystack.lower()
        include = task_include_keywords if case_sensitive else [k.lower() for k in task_include_keywords]
        exclude = task_exclude_keywords if case_sensitive else [k.lower() for k in task_exclude_keywords]

        if include and not any(k in haystack for k in include):
            return False
        if exclude and any(k in haystack for k in exclude):
            return False
        return True

    def _discover_episode_descriptors(
        self,
        resolved_zip_paths: list[Path],
        task_names: list[str] | None,
        task_categories: list[str] | None,
        robot_types: list[str] | None,
        task_include_keywords: list[str] | None,
        task_exclude_keywords: list[str] | None,
        task_keyword_case_sensitive: bool,
    ) -> list[_EpisodeDescriptor]:
        descriptors: list[_EpisodeDescriptor] = []

        total_before_filters = 0
        skipped_bad_zips: list[str] = []
        for zip_path in resolved_zip_paths:
            try:
                extract_dir = self._extract_zip_if_needed(zip_path)
            except (zipfile.BadZipFile, OSError, ValueError) as exc:
                logging.warning("Skipping invalid zip file '%s': %s", zip_path, exc)
                skipped_bad_zips.append(str(zip_path))
                continue
            json_paths = sorted(extract_dir.rglob("data.json"))
            total_before_filters += len(json_paths)

            for json_path in json_paths:
                try:
                    with open(json_path) as f:
                        frames = json.load(f)
                except (OSError, json.JSONDecodeError):
                    logging.warning("Skipping malformed episode json: %s", json_path)
                    continue

                if not isinstance(frames, list) or len(frames) == 0:
                    continue

                episode_dir = json_path.parent
                task_name, task_full_name, category_name = self._derive_task_from_episode_dir(
                    episode_dir=episode_dir,
                    extract_root=extract_dir,
                    zip_path=zip_path,
                )
                robot_type = _infer_robot_type(frames[0])

                descriptor = _EpisodeDescriptor(
                    json_path=json_path,
                    episode_dir=episode_dir,
                    task_name=task_name,
                    task_full_name=task_full_name,
                    category_name=category_name,
                    robot_type=robot_type,
                    length=len(frames),
                )

                if not self._episode_matches_filters(
                    descriptor=descriptor,
                    task_names=task_names,
                    task_categories=task_categories,
                    robot_types=robot_types,
                    task_include_keywords=task_include_keywords,
                    task_exclude_keywords=task_exclude_keywords,
                    case_sensitive=task_keyword_case_sensitive,
                ):
                    continue

                descriptors.append(descriptor)

        if skipped_bad_zips:
            logging.warning(
                "Skipped %d invalid zip files while building dataset: %s",
                len(skipped_bad_zips),
                skipped_bad_zips,
            )

        logging.info(
            "Zip episode discovery and filtering: total=%d -> matched=%d "
            "(task_names=%s, task_categories=%s, robot_types=%s, include=%s, exclude=%s, case_sensitive=%s)",
            total_before_filters,
            len(descriptors),
            task_names or [],
            task_categories or [],
            robot_types or [],
            task_include_keywords or [],
            task_exclude_keywords or [],
            task_keyword_case_sensitive,
        )
        return descriptors

    def _materialize_payload(
        self, descriptors: list[_EpisodeDescriptor]
    ) -> tuple[
        list[_EpisodePayload],
        list[int],
        list[int],
        int,
        int,
        tuple[int, int],
        list[str],
        dict[str, int],
        pd.DataFrame,
        datasets.Dataset,
        dict[str, dict[str, torch.Tensor]],
    ]:
        state_dim = 0
        action_dim = 0
        extra_observation_dims: dict[str, int] = defaultdict(int)
        available_camera_keys: set[str] = set()

        for desc in descriptors:
            with open(desc.json_path) as f:
                frames = json.load(f)
            for frame in frames:
                state_dim = max(state_dim, len(_extract_state_vector(frame)))
                action_dim = max(action_dim, len(_extract_action_vector(frame, desc.robot_type)))

                for camera_key, camera_path in _extract_camera_paths(frame).items():
                    if camera_path:
                        available_camera_keys.add(camera_key)

                for obs_key, values in _extract_extra_observation_vectors(frame).items():
                    extra_observation_dims[obs_key] = max(extra_observation_dims[obs_key], len(values))

        if state_dim == 0 or action_dim == 0:
            raise ValueError(
                "Could not infer state/action dimensions from zip dataset. "
                "Check that frames contain states.arm_state/hand_state and actions.sol_q/left_angles/right_angles."
            )

        camera_keys: list[str] = []
        for key in DEFAULT_CAMERA_PRIORITY:
            if key in available_camera_keys:
                camera_keys.append(key)
        for key in sorted(available_camera_keys):
            if key not in camera_keys:
                camera_keys.append(key)

        if not camera_keys:
            raise ValueError(
                "No camera streams were found in zip dataset frames. "
                "Expected at least one of: image/head_rgb/head_depth/*wrist*."
            )

        missing_wrist = sorted(WRIST_CAMERA_KEYS - set(camera_keys))
        if missing_wrist:
            logging.warning(
                "Wrist RGB/depth cameras were not found in Humanoid-Everyday zip frames. "
                "Proceeding without wrist cameras. Missing=%s, available=%s",
                missing_wrist,
                camera_keys,
            )

        task_order: dict[str, int] = {}
        for desc in descriptors:
            if desc.task_full_name not in task_order:
                task_order[desc.task_full_name] = len(task_order)
        tasks_df = pd.DataFrame({"task_index": list(task_order.values())}, index=list(task_order.keys()))

        state_sum = np.zeros(state_dim, dtype=np.float64)
        state_sumsq = np.zeros(state_dim, dtype=np.float64)
        state_min = np.full(state_dim, np.inf, dtype=np.float64)
        state_max = np.full(state_dim, -np.inf, dtype=np.float64)

        action_sum = np.zeros(action_dim, dtype=np.float64)
        action_sumsq = np.zeros(action_dim, dtype=np.float64)
        action_min = np.full(action_dim, np.inf, dtype=np.float64)
        action_max = np.full(action_dim, -np.inf, dtype=np.float64)

        extra_sum: dict[str, np.ndarray] = {}
        extra_sumsq: dict[str, np.ndarray] = {}
        extra_min: dict[str, np.ndarray] = {}
        extra_max: dict[str, np.ndarray] = {}
        for key, dim in extra_observation_dims.items():
            if dim <= 0:
                continue
            extra_sum[key] = np.zeros(dim, dtype=np.float64)
            extra_sumsq[key] = np.zeros(dim, dtype=np.float64)
            extra_min[key] = np.full(dim, np.inf, dtype=np.float64)
            extra_max[key] = np.full(dim, -np.inf, dtype=np.float64)

        payload: list[_EpisodePayload] = []
        episode_indices: list[int] = []
        dataset_from_index: list[int] = []
        dataset_to_index: list[int] = []
        episode_lengths: list[int] = []
        episode_tasks: list[list[str]] = []
        episode_robot_types: list[str] = []

        cursor = 0
        image_hw: tuple[int, int] | None = None

        for episode_index, desc in enumerate(descriptors):
            with open(desc.json_path) as f:
                frames = json.load(f)

            length = len(frames)
            states = np.zeros((length, state_dim), dtype=np.float32)
            actions = np.zeros((length, action_dim), dtype=np.float32)
            timestamps = np.zeros((length,), dtype=np.float32)
            camera_paths: dict[str, list[str]] = {key: [] for key in camera_keys}
            extra_observations: dict[str, np.ndarray] = {
                key: np.zeros((length, dim), dtype=np.float32)
                for key, dim in extra_observation_dims.items()
                if dim > 0
            }

            for frame_idx, frame in enumerate(frames):
                state_vec = np.asarray(_extract_state_vector(frame), dtype=np.float32)
                action_vec = np.asarray(_extract_action_vector(frame, desc.robot_type), dtype=np.float32)

                if state_vec.size > 0:
                    d = min(state_dim, state_vec.size)
                    states[frame_idx, :d] = state_vec[:d]
                if action_vec.size > 0:
                    d = min(action_dim, action_vec.size)
                    actions[frame_idx, :d] = action_vec[:d]

                frame_camera_paths = _extract_camera_paths(frame)
                for camera_key in camera_keys:
                    camera_paths[camera_key].append(str(frame_camera_paths.get(camera_key, "")))

                frame_extra_observations = _extract_extra_observation_vectors(frame)
                for obs_key, obs_tensor in extra_observations.items():
                    values = np.asarray(frame_extra_observations.get(obs_key, []), dtype=np.float32)
                    if values.size <= 0:
                        continue
                    d = min(obs_tensor.shape[1], values.size)
                    obs_tensor[frame_idx, :d] = values[:d]

                timestamps[frame_idx] = float(frame.get("time", frame_idx / self.fps))

            if image_hw is None and length > 0:
                for camera_key in camera_keys:
                    candidate_path = camera_paths[camera_key][0]
                    if not candidate_path:
                        continue
                    fpath = desc.episode_dir / candidate_path
                    if not fpath.exists():
                        continue
                    try:
                        if camera_key.endswith("_depth"):
                            with lzma.open(fpath, "rb") as f:
                                depth = np.asarray(np.load(f))
                            if depth.ndim >= 2:
                                image_hw = (int(depth.shape[0]), int(depth.shape[1]))
                        else:
                            with Image.open(fpath) as img:
                                img = img.convert("RGB")
                                image_hw = (img.height, img.width)
                    except Exception:
                        continue
                    if image_hw is not None:
                        break

            state_sum += states.sum(axis=0, dtype=np.float64)
            state_sumsq += np.square(states, dtype=np.float64).sum(axis=0, dtype=np.float64)
            state_min = np.minimum(state_min, states.min(axis=0, initial=np.inf))
            state_max = np.maximum(state_max, states.max(axis=0, initial=-np.inf))

            action_sum += actions.sum(axis=0, dtype=np.float64)
            action_sumsq += np.square(actions, dtype=np.float64).sum(axis=0, dtype=np.float64)
            action_min = np.minimum(action_min, actions.min(axis=0, initial=np.inf))
            action_max = np.maximum(action_max, actions.max(axis=0, initial=-np.inf))

            for obs_key, obs_tensor in extra_observations.items():
                if obs_tensor.shape[1] <= 0:
                    continue
                extra_sum[obs_key] += obs_tensor.sum(axis=0, dtype=np.float64)
                extra_sumsq[obs_key] += np.square(obs_tensor, dtype=np.float64).sum(axis=0, dtype=np.float64)
                extra_min[obs_key] = np.minimum(extra_min[obs_key], obs_tensor.min(axis=0, initial=np.inf))
                extra_max[obs_key] = np.maximum(extra_max[obs_key], obs_tensor.max(axis=0, initial=-np.inf))

            from_idx = cursor
            to_idx = cursor + length
            cursor = to_idx

            payload.append(
                _EpisodePayload(
                    episode_index=episode_index,
                    episode_dir=desc.episode_dir,
                    task_name=desc.task_full_name,
                    robot_type=desc.robot_type,
                    dataset_from_index=from_idx,
                    dataset_to_index=to_idx,
                    camera_paths=camera_paths,
                    states=states,
                    extra_observations=extra_observations,
                    actions=actions,
                    timestamps=timestamps,
                )
            )

            episode_indices.append(episode_index)
            dataset_from_index.append(from_idx)
            dataset_to_index.append(to_idx)
            episode_lengths.append(length)
            episode_tasks.append([desc.task_full_name])
            episode_robot_types.append(desc.robot_type)

        if cursor == 0:
            raise ValueError("No frames available after loading zip episodes.")

        state_mean = (state_sum / cursor).astype(np.float32)
        state_var = np.maximum(state_sumsq / cursor - np.square(state_mean, dtype=np.float64), 1e-12)
        state_std = np.sqrt(state_var).astype(np.float32)

        action_mean = (action_sum / cursor).astype(np.float32)
        action_var = np.maximum(action_sumsq / cursor - np.square(action_mean, dtype=np.float64), 1e-12)
        action_std = np.sqrt(action_var).astype(np.float32)

        stats: dict[str, dict[str, torch.Tensor]] = {
            "observation.state": {
                "mean": torch.from_numpy(state_mean),
                "std": torch.from_numpy(state_std),
                "min": torch.from_numpy(state_min.astype(np.float32)),
                "max": torch.from_numpy(state_max.astype(np.float32)),
            },
            "action": {
                "mean": torch.from_numpy(action_mean),
                "std": torch.from_numpy(action_std),
                "min": torch.from_numpy(action_min.astype(np.float32)),
                "max": torch.from_numpy(action_max.astype(np.float32)),
            },
        }
        for camera_key in camera_keys:
            stats[camera_key] = {}

        for obs_key, dim in extra_observation_dims.items():
            if dim <= 0 or obs_key not in extra_sum:
                continue
            obs_mean = (extra_sum[obs_key] / cursor).astype(np.float32)
            obs_var = np.maximum(extra_sumsq[obs_key] / cursor - np.square(obs_mean, dtype=np.float64), 1e-12)
            obs_std = np.sqrt(obs_var).astype(np.float32)
            stats[obs_key] = {
                "mean": torch.from_numpy(obs_mean),
                "std": torch.from_numpy(obs_std),
                "min": torch.from_numpy(extra_min[obs_key].astype(np.float32)),
                "max": torch.from_numpy(extra_max[obs_key].astype(np.float32)),
            }

        episodes_ds = datasets.Dataset.from_dict(
            {
                "episode_index": episode_indices,
                "dataset_from_index": dataset_from_index,
                "dataset_to_index": dataset_to_index,
                "length": episode_lengths,
                "tasks": episode_tasks,
                "robot_type": episode_robot_types,
            }
        )

        if image_hw is None:
            image_hw = (480, 640)

        return (
            payload,
            dataset_from_index,
            dataset_to_index,
            state_dim,
            action_dim,
            image_hw,
            camera_keys,
            dict(extra_observation_dims),
            tasks_df,
            episodes_ds,
            stats,
        )

    @staticmethod
    def _infer_global_robot_type(descriptors: list[_EpisodeDescriptor]) -> str:
        robot_types = sorted({d.robot_type for d in descriptors})
        if len(robot_types) == 1:
            return robot_types[0]
        return "mixed"

    @staticmethod
    def _build_features(
        state_dim: int,
        action_dim: int,
        image_shape_hw: tuple[int, int],
        camera_keys: list[str],
        extra_observation_dims: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        h, w = image_shape_hw
        features: dict[str, dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"action_{i}" for i in range(action_dim)],
            },
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
            "index": {"dtype": "int64", "shape": (1,), "names": None},
            "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        }

        for camera_key in camera_keys:
            features[camera_key] = {
                "dtype": "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            }

        for obs_key, dim in sorted(extra_observation_dims.items()):
            if dim <= 0:
                continue
            features[obs_key] = {
                "dtype": "float32",
                "shape": (dim,),
                "names": [f"{obs_key.replace('.', '_')}_{i}" for i in range(dim)],
            }

        return features

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return len(self._episodes_payload)

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.meta.features

    def __len__(self) -> int:
        return self.num_frames

    def _get_episode_and_local_index(self, idx: int) -> tuple[_EpisodePayload, int]:
        if idx < 0:
            idx += self.num_frames
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index out of bounds: {idx}")

        episode_idx = bisect.bisect_right(self._dataset_to_indices, idx)
        episode = self._episodes_payload[episode_idx]
        local_idx = idx - episode.dataset_from_index
        return episode, local_idx

    def _compute_delta_query_indices(
        self, abs_idx: int, episode: _EpisodePayload
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        if self.delta_indices is None:
            return {}, {}

        ep_start = episode.dataset_from_index
        ep_end = episode.dataset_to_index
        query_indices: dict[str, list[int]] = {}
        padding: dict[str, torch.Tensor] = {}

        for key, deltas in self.delta_indices.items():
            query_indices[key] = [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in deltas]
            padding[f"{key}_is_pad"] = torch.BoolTensor(
                [(abs_idx + delta < ep_start) or (abs_idx + delta >= ep_end) for delta in deltas]
            )

        return query_indices, padding

    def _zero_image_tensor(self) -> torch.Tensor:
        h, w = self._image_shape_hw
        return torch.zeros((3, h, w), dtype=torch.float32)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return self._to_tensor(img)

    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        with lzma.open(depth_path, "rb") as f:
            depth_bytes = f.read()

        depth: np.ndarray | None = None
        try:
            depth = np.asarray(np.load(io.BytesIO(depth_bytes), allow_pickle=False), dtype=np.float32)
        except Exception:
            h, w = self._image_shape_hw
            expected_pixels = h * w
            if len(depth_bytes) == expected_pixels * 2:
                depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(h, w).astype(np.float32)
            elif len(depth_bytes) == expected_pixels * 4:
                depth = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w).astype(np.float32)
            else:
                return self._zero_image_tensor()

        if depth.ndim >= 3:
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            return self._zero_image_tensor()

        finite_mask = np.isfinite(depth)
        depth_norm = np.zeros_like(depth, dtype=np.float32)
        if finite_mask.any():
            finite_values = depth[finite_mask]
            lo = np.percentile(finite_values, 1)
            hi = np.percentile(finite_values, 99)
            denom = max(float(hi - lo), 1e-6)
            depth_norm[finite_mask] = np.clip((depth[finite_mask] - lo) / denom, 0.0, 1.0)

        return torch.from_numpy(np.repeat(depth_norm[None, :, :], 3, axis=0))

    def _load_camera_tensor(self, camera_key: str, fpath: Path) -> torch.Tensor:
        if not fpath.exists():
            if camera_key not in self._missing_modalities_warned:
                logging.warning("Camera file not found for %s: %s (using zeros)", camera_key, fpath)
                self._missing_modalities_warned.add(camera_key)
            return self._zero_image_tensor()

        try:
            if camera_key.endswith("_depth"):
                return self._load_depth(fpath)
            return self._load_image(fpath)
        except Exception as exc:
            if camera_key not in self._missing_modalities_warned:
                logging.warning(
                    "Failed to load camera file for %s: %s (%s). Using zeros.",
                    camera_key,
                    fpath,
                    exc,
                )
                self._missing_modalities_warned.add(camera_key)
            return self._zero_image_tensor()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        episode, local_idx = self._get_episode_and_local_index(idx)
        task_idx = int(self.meta.tasks.loc[episode.task_name, "task_index"])

        camera_tensors: dict[str, torch.Tensor] = {}
        for camera_key, rel_paths in episode.camera_paths.items():
            rel_path = rel_paths[local_idx] if local_idx < len(rel_paths) else ""
            fpath = episode.episode_dir / rel_path if rel_path else Path("")
            camera_tensor = self._load_camera_tensor(camera_key, fpath) if rel_path else self._zero_image_tensor()
            if self.image_transforms is not None and not camera_key.endswith("_depth"):
                camera_tensor = self.image_transforms(camera_tensor)
            camera_tensors[camera_key] = camera_tensor

        item: dict[str, Any] = {
            "observation.state": torch.from_numpy(episode.states[local_idx]),
            "action": torch.from_numpy(episode.actions[local_idx]),
            "timestamp": torch.tensor(episode.timestamps[local_idx], dtype=torch.float32),
            "frame_index": torch.tensor(local_idx, dtype=torch.int64),
            "episode_index": torch.tensor(episode.episode_index, dtype=torch.int64),
            "index": torch.tensor(idx, dtype=torch.int64),
            "task_index": torch.tensor(task_idx, dtype=torch.int64),
            "task": episode.task_name,
        }
        item.update(camera_tensors)

        for obs_key, obs_values in episode.extra_observations.items():
            item[obs_key] = torch.from_numpy(obs_values[local_idx])

        query_indices, padding = self._compute_delta_query_indices(idx, episode)
        item.update(padding)
        for key, abs_query_indices in query_indices.items():
            local_query_indices = [q - episode.dataset_from_index for q in abs_query_indices]
            if key == "action":
                item["action"] = torch.from_numpy(episode.actions[local_query_indices])
            elif key == "observation.state":
                item["observation.state"] = torch.from_numpy(episode.states[local_query_indices])
            else:
                if key not in self._unsupported_delta_keys:
                    logging.warning(
                        "Delta query key '%s' is not implemented for HumanoidEverydayZipDataset; skipping.",
                        key,
                    )
                    self._unsupported_delta_keys.add(key)

        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  repo_id='{self.repo_id}',\n"
            f"  num_episodes={self.num_episodes},\n"
            f"  num_frames={self.num_frames},\n"
            f"  features={list(self.features.keys())},\n"
            ")"
        )
