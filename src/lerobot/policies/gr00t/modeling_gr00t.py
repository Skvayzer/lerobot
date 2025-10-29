# === IsaacGr00t START ===
from __future__ import annotations

import math
from collections import deque
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import Gr00tPolicy

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues

from .configuration_gr00t import IsaacGr00tConfig


class IsaacGr00tPolicy(PreTrainedPolicy):
    config_class = IsaacGr00tConfig
    name = "isaac_gr00t"

    def __init__(self, config: IsaacGr00tConfig, dataset_stats: dict | None = None):
        super().__init__(config)
        config.validate_features()

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self._device = torch.device(config.device)
        self._motor_order = list(config.motor_order)
        self._state_groups = dict(config.state_groups)
        self._state_dim = len(self._motor_order)
        self._chunk_size = config.chunk_size
        self._obs_steps = config.n_obs_steps

        (
            self._modality_config,
            self._modality_transform,
            self._state_key_to_indices,
            self._action_key_to_indices,
            self._video_keys,
            self._language_keys,
        ) = self._build_gr00t_data_pipeline(config)
        self._preferred_language_key = (
            config.language_key
            or (self._language_keys[0] if self._language_keys else "annotation.human.action.task_description")
        )

        self._gr00t_policy = Gr00tPolicy(
            model_path=config.base_model_path,
            embodiment_tag=config.embodiment_tag,
            modality_config=self._modality_config,
            modality_transform=self._modality_transform,
            denoising_steps=config.denoising_steps,
            device=config.device,
        )
        self.model = self._gr00t_policy.model
        self.reset()

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        self.train()
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        prepared = self._prepare_gr00t_batch(batch, include_action=True, training=True)
        outputs = self.model(prepared)
        loss = outputs["loss"]
        metrics = {"loss": loss.detach()}
        return loss, metrics

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        prepared = self._prepare_gr00t_batch(batch, include_action=False, training=False)
        if noise is not None:
            noise = noise.to(self._device)
        predictions = self.model.get_action(prepared)["action_pred"]
        predictions = predictions.to(torch.float32)
        actions = self._convert_predictions_to_actions(predictions)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        batch = self.normalize_inputs(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _build_gr00t_data_pipeline(
        self, config: IsaacGr00tConfig
    ) -> tuple[
        dict[str, ModalityConfig],
        Any,
        dict[str, list[int]],
        dict[str, list[int]],
        list[str],
        list[str],
    ]:
        data_cfg = load_data_config(config.data_config)
        modality_config = data_cfg.modality_config()
        transform = data_cfg.transform()
        transform.eval()

        state_key_to_indices: dict[str, list[int]] = {}
        for key in getattr(data_cfg, "state_keys", []):
            _, group = key.split(".", 1)
            if group not in self._state_groups:
                raise ValueError(f"State key '{group}' missing from config.state_groups.")
            state_key_to_indices[key] = self._state_groups[group]

        action_key_to_indices: dict[str, list[int]] = {}
        for key in getattr(data_cfg, "action_keys", []):
            _, group = key.split(".", 1)
            if group not in self._state_groups:
                raise ValueError(f"Action key '{group}' missing from config.state_groups.")
            action_key_to_indices[key] = self._state_groups[group]

        video_keys = list(getattr(data_cfg, "video_keys", ["video.ego_view"]))
        language_keys = list(getattr(data_cfg, "language_keys", []))

        return (
            modality_config,
            transform,
            state_key_to_indices,
            action_key_to_indices,
            video_keys,
            language_keys,
        )

    def _prepare_gr00t_batch(
        self,
        batch: dict[str, Tensor],
        *,
        include_action: bool,
        training: bool,
    ) -> dict[str, Any]:
        raw = {}
        camera_array = self._extract_video(batch)
        raw.update(camera_array)

        if OBS_STATE not in batch:
            raise KeyError(f"State key '{OBS_STATE}' missing from batch.")
        state_arrays = self._extract_group_arrays(batch.get(OBS_STATE), self._state_key_to_indices, self._obs_steps)
        raw.update(state_arrays)

        if include_action and ACTION in batch:
            action_arrays = self._extract_group_arrays(batch[ACTION], self._action_key_to_indices, self._chunk_size)
            raw.update(action_arrays)
        elif include_action:
            raise KeyError(f"Action key '{ACTION}' missing from batch.")

        language = self._extract_language(batch)
        raw.update(language)

        processed = self._apply_modality_transform(raw, training=training)
        return processed

    def _apply_modality_transform(self, data: dict[str, Any], *, training: bool) -> dict[str, Any]:
        if training:
            self._modality_transform.train()
        else:
            self._modality_transform.eval()
        transformed = self._modality_transform(data)
        for key, value in transformed.items():
            if isinstance(value, torch.Tensor):
                transformed[key] = value.to(self._device)
            elif isinstance(value, np.ndarray):
                transformed[key] = torch.from_numpy(value).to(self._device)
        return transformed

    def _extract_video(self, batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
        cam_key = self.config.camera_key
        if cam_key not in batch:
            raise KeyError(f"Camera key '{cam_key}' missing from batch.")
        video = batch[cam_key]
        video = video.detach().cpu()
        if video.ndim == 4:
            # [B, H, W, C] or [B, C, H, W]
            if video.shape[-1] == 3:
                video = video[:, None, ...]
            else:
                video = video.unsqueeze(1).permute(0, 1, 3, 4, 2)
        elif video.ndim == 5:
            if video.shape[-1] != 3:
                video = video.permute(0, 1, 3, 4, 2)
        else:
            raise ValueError(f"Unexpected video tensor shape: {tuple(video.shape)}")

        np_video = video.numpy()
        if np_video.dtype != np.uint8:
            np_video = np.clip(np_video * 255.0, 0.0, 255.0).astype(np.uint8)

        video_key = self._video_keys[0]
        return {video_key: np_video}

    def _extract_group_arrays(
        self,
        tensor: Tensor | None,
        key_to_indices: dict[str, list[int]],
        horizon: int,
    ) -> dict[str, np.ndarray]:
        if tensor is None:
            return {}
        data = tensor.detach().cpu()
        if data.ndim == 2:
            data = data.unsqueeze(1)
        if data.shape[1] < horizon:
            repeat_times = math.ceil(horizon / data.shape[1])
            data = data.repeat(1, repeat_times, 1)
        data = data[:, :horizon, :]
        arrays = {}
        for key, indices in key_to_indices.items():
            arrays[key] = data[..., indices].numpy()
        return arrays

    def _extract_language(self, batch: dict[str, Tensor]) -> dict[str, list[str]]:
        key = self._preferred_language_key
        if key in batch:
            entries = batch[key]
            if isinstance(entries, Tensor):
                texts = [str(entry) for entry in entries]
            elif isinstance(entries, Iterable) and not isinstance(entries, (str, bytes)):
                texts = [str(entry) for entry in entries]
            else:
                size = self._infer_batch_size(batch)
                texts = [str(entries)] * size
        else:
            size = self._infer_batch_size(batch)
            texts = [self.config.default_language] * size
        return {key: texts}

    def _infer_batch_size(self, batch: dict[str, Any]) -> int:
        for value in batch.values():
            if isinstance(value, Tensor):
                return value.shape[0]
            if isinstance(value, np.ndarray):
                return value.shape[0]
        raise ValueError("Unable to infer batch size from batch values.")

    def _convert_predictions_to_actions(self, predictions: Tensor) -> Tensor:
        predictions_cpu = predictions.detach().cpu()
        unnormalized = self._modality_transform.unapply({"action": predictions_cpu})
        batch_size = predictions_cpu.shape[0]
        actions = np.zeros((batch_size, predictions_cpu.shape[1], self._state_dim), dtype=np.float32)
        for key, indices in self._action_key_to_indices.items():
            values = unnormalized[key]
            actions[..., indices] = values[..., : len(indices)]
        actions_tensor = torch.from_numpy(actions).to(self._device)
        actions_tensor = self.unnormalize_outputs({ACTION: actions_tensor})[ACTION]
        return actions_tensor
# === IsaacGr00t END ===
