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

"""
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootPolicy.finetune_with_groot_runner` below.
"""

import os
import threading
import time
from collections import deque
from typing import Any

import torch
from torch import Tensor
from transformers import BatchFeature

from lerobot.inference.guidance import apply_cfg_guidance_velocity
from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
from lerobot.policies.grootCoT.groot_n1 import GR00TN15
from lerobot.policies.pretrained import PreTrainedPolicy


class GrootCoTPolicy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    name = "groot_cot"
    config_class = GrootCoTConfig

    def __init__(self, config: GrootCoTConfig, dataset_stats: dict | None = None, **kwargs):
        """Initialize Groot policy wrapper."""
        super().__init__(config, **kwargs)
        config.validate_features()
        self.config = config

        # Initialize GR00T model using ported components
        self._groot_model = self._create_groot_model()

        self.reset()

    def __del__(self):
        try:
            self._shutdown_async_worker()
        except Exception:
            pass

    def _create_groot_model(self):
        """Create and initialize the GR00T model using Isaac-GR00T API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to Isaac-GR00T):
        1) Download and load pretrained model via GR00TN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        train_projector_only = getattr(self.config, "train_vlm_projector_only", False)
        tune_llm = False if train_projector_only else self.config.tune_llm
        tune_visual = False if train_projector_only else self.config.tune_visual
        # If training projector only, keep action-head projector trainable but freeze diffusion.
        tune_projector = True if train_projector_only else self.config.tune_projector
        tune_diffusion_model = False if train_projector_only else self.config.tune_diffusion_model
        tune_vlm_projector = True if train_projector_only else self.config.tune_vlm_projector
        tune_vlln = False if train_projector_only else self.config.tune_vlln
        tune_top_llm_layers = 0 if train_projector_only else self.config.tune_top_llm_layers

        extra_observation_dims = self._get_extra_observation_dims()
        system2_preset = getattr(self.config, "system2_vlm_preset", "unknown")
        system2_source = getattr(self.config, "resolved_system2_vlm_source", "legacy_vlm_processor_model_id")
        system2_model_id = getattr(
            self.config,
            "resolved_system2_vlm_model_id",
            self.config.vlm_processor_model_id,
        )
        print(
            "[GROOT][System2] "
            f"preset={system2_preset}, "
            f"source={system2_source}, "
            f"model_id={system2_model_id}"
        )

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlm_projector=tune_vlm_projector,
            tune_vlln=tune_vlln,
            tune_top_llm_layers=tune_top_llm_layers,
            model_id=system2_model_id,
            attn_implementation=getattr(self.config, "attn_implementation", None),
            # LoRA for Backbone
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=self.config.lora_target_modules,
            # LoRA for Action Head
            action_head_lora_rank=self.config.action_head_lora_rank,
            action_head_lora_alpha=self.config.action_head_lora_alpha,
            action_head_lora_dropout=self.config.action_head_lora_dropout,
            action_head_lora_target_modules=self.config.action_head_lora_target_modules,
            # Precision
            load_bf16=self.config.use_bf16,
            torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
            # Optional extra observation projectors for action head
            extra_observation_dims=extra_observation_dims,
            # Optional weighted action-group split.
            primary_action_group_indices=self.config.primary_action_group_indices,
            secondary_action_group_indices=self.config.secondary_action_group_indices,
            primary_action_group_loss_weight=self.config.primary_action_group_loss_weight,
            secondary_action_group_loss_weight=self.config.secondary_action_group_loss_weight,
            # Legacy aliases kept for compatibility with older configs.
            upper_body_joint_indices=self.config.upper_body_joint_indices,
            lower_body_joint_indices=self.config.lower_body_joint_indices,
            upper_body_loss_weight=self.config.upper_body_loss_weight,
            lower_body_loss_weight=self.config.lower_body_loss_weight,
            # N1.6 action head
            action_head_version=getattr(self.config, "action_head_version", "n15"),
            n16_action_head_weights_path=getattr(self.config, "n16_action_head_weights_path", None),
            chunk_size=self.config.chunk_size,
            # IK prior
            ik_prior_prob=getattr(self.config, "ik_prior_prob", 0.0),
            ik_prior_noise_scale=getattr(self.config, "ik_prior_noise_scale", 0.15),
            ik_prior_arm_dim=getattr(self.config, "ik_prior_arm_dim", 14),
            # Optional RECAP/value extensions.
            value_head_enable=bool(getattr(self.config, "recap_value_head_enable", False)),
            tune_value_head=bool(getattr(self.config, "recap_tune_value_head", True)),
            value_head_bins=int(getattr(self.config, "recap_value_head_bins", 201)),
            value_head_vmin=float(getattr(self.config, "recap_value_head_vmin", -1.0)),
            value_head_vmax=float(getattr(self.config, "recap_value_head_vmax", 0.0)),
            value_head_pooling=str(getattr(self.config, "recap_value_head_pooling", "masked_mean")),
        )

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        return model

    def _get_extra_observation_dims(self) -> dict[str, int]:
        extra_dims: dict[str, int] = {}
        for key, feat in (self.config.input_features or {}).items():
            if not key.startswith("observation.extra."):
                continue
            shape = getattr(feat, "shape", None)
            if not shape:
                continue
            try:
                dim = int(shape[-1])
            except (TypeError, ValueError):
                continue
            if dim > 0:
                extra_dims[key] = dim

        if getattr(self.config, "recap_enable", False):
            indicator_key = getattr(self.config, "recap_adv_indicator_key", "observation.extra.adv_indicator")
            extra_dims.setdefault(indicator_key, 1)

        if extra_dims:
            print(f"[GROOT] Extra observation projections enabled for keys: {sorted(extra_dims.keys())}")
        return extra_dims

    def reset(self):
        """Reset policy state when environment resets."""
        self._shutdown_async_worker()
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._cached_backbone_outputs: BatchFeature | None = None
        self._cached_train_backbone_outputs: BatchFeature | None = None
        self._last_backbone_refresh_inference_step: int | None = None
        self._last_backbone_refresh_train_step: int | None = None
        self._last_replan_step: int | None = None
        self._inference_step: int = 0
        self._train_forward_step: int = 0
        self._system2_refresh_count_inference: int = 0
        self._system2_refresh_count_train: int = 0
        self._inference_start_time: float = time.perf_counter()
        self._s1_fallback_count: int = 0
        self._last_emitted_action: Tensor | None = None
        self._last_fallback_action: Tensor | None = None
        self._async_startup_warmup_done: bool = False
        self._async_step_interval_warned: bool = False
        self._async_last_error_log_step: int = -1
        self._async_lock = threading.Lock()
        self._async_stop_event = threading.Event()
        self._async_new_obs_event = threading.Event()
        self._async_thread: threading.Thread | None = None
        self._async_latest_inputs: dict[str, Tensor] | None = None
        self._async_latest_inputs_step: int | None = None
        self._async_latest_inputs_time: float | None = None
        self._async_ready_chunks: deque[Tensor] = deque(
            [], maxlen=max(1, int(getattr(self.config, "system2_async_prefetch_chunks", 1)))
        )
        self._async_worker_busy: bool = False
        self._async_last_refresh_time: float | None = None
        self._async_refresh_count: int = 0
        self._async_exception: str | None = None

        # CraftNet System 2 → System 1 semantic intent state
        self._current_subtask_text: str | None = None
        self._current_target_bbox: list[float] | None = None
        self._subtask_index: int = 0

    def get_optim_params(self) -> dict:
        return self.parameters()

    @staticmethod
    def _build_groot_inputs(batch: dict[str, Tensor], include_action: bool) -> dict[str, Tensor]:
        """Filter a preprocessed batch to keys consumed by GR00T.

        NOTE: qwen_pixel_values and qwen_image_grid_thw are consumed by both
        run_backbone (full Qwen forward) and run_visual_only (ViT-only fast path).
        These keys must not be stripped by any intermediate processing step.
        """
        allowed_base = {"state", "state_mask", "embodiment_id"}
        if include_action:
            allowed_base.update({"action", "action_mask"})
        return {
            k: v
            for k, v in batch.items()
            if (
                k in allowed_base
                or k.startswith("qwen_")
                or k.startswith("eagle_")  # backward compatibility for older checkpoints
                or k.startswith("observation.extra.")
            )
            and not (k.startswith("next.") or k == "info")
        }

    @staticmethod
    def _detach_batch_feature(batch_feature: BatchFeature) -> BatchFeature:
        detached = {}
        for key, value in batch_feature.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
            else:
                detached[key] = value
        return BatchFeature(data=detached)

    @staticmethod
    def _clone_batch_feature(batch_feature: BatchFeature) -> BatchFeature:
        cloned = {}
        for key, value in batch_feature.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                cloned[key] = value
        return BatchFeature(data=cloned)

    def _recap_indicator_key(self) -> str:
        return str(getattr(self.config, "recap_adv_indicator_key", "observation.extra.adv_indicator"))

    @staticmethod
    def _infer_batch_shape_dtype(batch: dict[str, Tensor]) -> tuple[int, torch.device, torch.dtype]:
        def _dtype_of(value: torch.Tensor) -> torch.dtype:
            return value.dtype if value.dtype.is_floating_point else torch.float32

        # Prefer explicit policy tensors where dim-0 is guaranteed to be batch.
        for key in ("state", "action", "state_mask", "action_mask", "embodiment_id"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim >= 2:
                    return int(value.shape[0]), value.device, _dtype_of(value)
                if value.ndim == 1:
                    # In inference we often get a single sample vector without batch dim.
                    return 1, value.device, _dtype_of(value)
                return 1, value.device, _dtype_of(value)

        # Token ids/masks usually carry an explicit batch dimension (B, S).
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.endswith("input_ids") or key.endswith("attention_mask"):
                if value.ndim >= 2:
                    return int(value.shape[0]), value.device, _dtype_of(value)
                return 1, value.device, _dtype_of(value)

        # Fallback for generic tensors, but avoid flattened visual tensors where dim-0 is token count.
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.ndim >= 2:
                if key.endswith("pixel_values") and value.ndim == 2:
                    continue
                if key.endswith("image_grid_thw") and value.ndim == 2:
                    continue
                return int(value.shape[0]), value.device, _dtype_of(value)

        # Last resort: single-sample assumption for 1D tensors.
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return 1, value.device, _dtype_of(value)
        return 1, torch.device("cpu"), torch.float32

    @staticmethod
    def _normalize_indicator_tensor(
        indicator: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ind = indicator.to(device=device, dtype=dtype)
        if ind.ndim == 0:
            ind = ind.repeat(batch_size).view(batch_size, 1)
        elif ind.ndim == 1:
            if ind.shape[0] == batch_size:
                ind = ind.unsqueeze(-1)
            else:
                ind = ind.reshape(batch_size, -1)[:, :1]
        elif ind.ndim == 2:
            if ind.shape[0] != batch_size:
                ind = ind.reshape(batch_size, -1)
            ind = ind[:, :1]
        else:
            if ind.shape[0] != batch_size:
                ind = ind.reshape(batch_size, -1)
            elif ind.shape[1] == 1:
                ind = ind[:, 0, :].reshape(batch_size, -1)
            else:
                ind = ind.reshape(batch_size, -1)
            ind = ind[:, :1]
        return ind

    def _set_indicator_tensor(
        self,
        batch: dict[str, Tensor],
        indicator_value: float | torch.Tensor,
    ) -> dict[str, Tensor]:
        key = self._recap_indicator_key()
        batch_size, device, dtype = self._infer_batch_shape_dtype(batch)
        indicator_tensor = torch.as_tensor(indicator_value, device=device, dtype=dtype)
        batch[key] = self._normalize_indicator_tensor(
            indicator_tensor,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return batch

    def _prepare_recap_indicator_batch(
        self,
        batch: dict[str, Tensor],
        *,
        apply_dropout: bool,
    ) -> dict[str, Tensor]:
        if not getattr(self.config, "recap_enable", False):
            return batch

        key = self._recap_indicator_key()
        null_value = float(getattr(self.config, "recap_adv_indicator_null_value", -1.0))
        out_batch = dict(batch)
        batch_size, device, dtype = self._infer_batch_shape_dtype(out_batch)

        if key not in out_batch:
            out_batch[key] = torch.full(
                (batch_size, 1),
                fill_value=null_value,
                device=device,
                dtype=dtype,
            )
        else:
            out_batch[key] = self._normalize_indicator_tensor(
                out_batch[key],
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        if not apply_dropout or not self.training:
            return out_batch

        dropout_p = float(getattr(self.config, "recap_adv_indicator_dropout_p", 0.0))
        if dropout_p <= 0.0:
            return out_batch

        drop_mask = torch.rand((batch_size, 1), device=device) < dropout_p
        indicator = out_batch[key]
        out_batch[key] = torch.where(
            drop_mask,
            torch.full_like(indicator, null_value),
            indicator,
        )
        return out_batch

    def _dual_rate_enabled(self) -> bool:
        return bool(getattr(self.config, "dual_rate_enable", False))

    def _async_dual_rate_enabled(self) -> bool:
        return self._dual_rate_enabled() and bool(getattr(self.config, "system2_async_enable", False))

    def _dual_rate_train_enabled(self) -> bool:
        return self._dual_rate_enabled() and bool(getattr(self.config, "dual_rate_apply_in_train", False))

    def _get_system2_async_period_s(self) -> float:
        system2_hz = max(float(getattr(self.config, "system2_hz", 1.0)), 1e-6)
        return 1.0 / system2_hz

    @staticmethod
    def _snapshot_groot_inputs_for_worker(groot_inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        snapshot: dict[str, Tensor] = {}
        for key, value in groot_inputs.items():
            if isinstance(value, torch.Tensor):
                snapshot[key] = value.detach().clone()
            else:
                snapshot[key] = value
        return snapshot

    def _publish_latest_inputs_for_worker(self, groot_inputs: dict[str, Tensor]) -> None:
        if not self._async_dual_rate_enabled():
            return
        snapshot = self._snapshot_groot_inputs_for_worker(groot_inputs)
        with self._async_lock:
            self._async_latest_inputs = snapshot
            self._async_latest_inputs_step = self._inference_step
            self._async_latest_inputs_time = time.time()
            self._async_new_obs_event.set()

    def _start_async_worker_if_needed(self) -> None:
        if not self._async_dual_rate_enabled():
            return
        with self._async_lock:
            if self._async_thread is not None and self._async_thread.is_alive():
                return
            self._async_stop_event.clear()
            self._async_new_obs_event.clear()
            self._async_exception = None
            self._async_thread = threading.Thread(
                target=self._async_system2_worker_loop,
                name="groot-system2-worker",
                daemon=True,
            )
            self._async_thread.start()
            if not bool(getattr(self.config, "system2_async_wall_clock", True)) and not self._async_step_interval_warned:
                print(
                    "[GROOT][ASYNC] system2_async_enable=true: step-based async scheduling is not supported; "
                    "using wall-clock scheduling from system2_hz."
                )
                self._async_step_interval_warned = True

    def _shutdown_async_worker(self) -> None:
        stop_event = getattr(self, "_async_stop_event", None)
        if stop_event is None:
            return
        stop_event.set()
        new_obs_event = getattr(self, "_async_new_obs_event", None)
        if new_obs_event is not None:
            new_obs_event.set()
        worker = getattr(self, "_async_thread", None)
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
        self._async_thread = None

    def _async_system2_worker_loop(self) -> None:
        period_s = self._get_system2_async_period_s()
        next_tick = time.monotonic()
        max_obs_age_s = float(getattr(self.config, "system2_async_max_observation_age_s", 0.5))
        while not self._async_stop_event.is_set():
            now = time.monotonic()
            timeout_s = max(0.0, next_tick - now)
            self._async_new_obs_event.wait(timeout=timeout_s)
            if self._async_stop_event.is_set():
                break
            # Consume wakeup and rely on latest snapshot only.
            self._async_new_obs_event.clear()
            now = time.monotonic()
            if now < next_tick:
                continue
            with self._async_lock:
                latest_inputs = self._async_latest_inputs
                latest_obs_time = self._async_latest_inputs_time
            if latest_inputs is None:
                next_tick = now + period_s
                continue
            if latest_obs_time is not None and (time.time() - latest_obs_time) > max_obs_age_s:
                next_tick = now + period_s
                continue

            with self._async_lock:
                self._async_worker_busy = True
            try:
                actions = self._compute_action_chunk_sync(
                    groot_inputs=latest_inputs,
                    force_backbone_refresh=True,
                    use_cached_dual_rate=False,
                )
                action_chunk = actions.transpose(0, 1).detach().clone()
                with self._async_lock:
                    self._async_ready_chunks.append(action_chunk)
                    self._async_refresh_count += 1
                    self._async_last_refresh_time = time.time()
                    self._async_exception = None
            except Exception as exc:  # pragma: no cover - defensive path
                with self._async_lock:
                    self._async_exception = repr(exc)
            finally:
                with self._async_lock:
                    self._async_worker_busy = False

            next_tick += period_s
            now = time.monotonic()
            while next_tick < now:
                next_tick += period_s

    def _pop_latest_async_chunk(self) -> Tensor | None:
        with self._async_lock:
            if len(self._async_ready_chunks) == 0:
                return None
            # Keep freshest chunk to maximize reactivity.
            latest_chunk = self._async_ready_chunks[-1]
            self._async_ready_chunks.clear()
            return latest_chunk

    def _state_based_fallback_action(self, batch: dict[str, Tensor]) -> Tensor:
        action_dim = int(self.config.output_features["action"].shape[0])

        def _extract_state_slice(state_tensor: torch.Tensor) -> Tensor:
            if state_tensor.ndim == 1:
                vec = state_tensor
            elif state_tensor.ndim == 2:
                vec = state_tensor[0]
            else:
                vec = state_tensor.reshape(state_tensor.shape[0], -1)[0]
            if vec.shape[0] < action_dim:
                out = torch.zeros((action_dim,), device=vec.device, dtype=vec.dtype)
                out[: vec.shape[0]] = vec
                vec = out
            else:
                vec = vec[:action_dim]
            return vec.unsqueeze(0)

        state_tensor = batch.get("state")
        if isinstance(state_tensor, torch.Tensor):
            return _extract_state_slice(state_tensor)
        legacy_state = batch.get("observation.state")
        if isinstance(legacy_state, torch.Tensor):
            return _extract_state_slice(legacy_state)

        if self._last_emitted_action is not None:
            return self._last_emitted_action.detach().clone()

        # Final fallback: zero action on model device.
        device = next(self.parameters()).device
        return torch.zeros((1, action_dim), device=device, dtype=torch.float32)

    def _maybe_log_async_error(self) -> None:
        if not self._async_dual_rate_enabled():
            return
        with self._async_lock:
            err = self._async_exception
        if err is None:
            return
        interval = max(1, int(getattr(self.config, "system2_async_log_every_n_steps", 100)))
        if self._async_last_error_log_step < 0 or (self._inference_step - self._async_last_error_log_step) >= interval:
            print(f"[GROOT][ASYNC] System-2 worker error: {err}")
            self._async_last_error_log_step = self._inference_step

    def _get_system2_interval_steps(self) -> int:
        if hasattr(self.config, "get_system2_update_interval_steps"):
            return int(self.config.get_system2_update_interval_steps())
        return 1

    @staticmethod
    def _should_refresh_cache(
        *,
        current_step: int,
        last_refresh_step: int | None,
        refresh_interval: int,
        force_refresh: bool,
        has_cache: bool,
    ) -> bool:
        if force_refresh or not has_cache:
            return True
        if last_refresh_step is None:
            return True
        return (current_step - last_refresh_step) >= max(1, refresh_interval)

    def _should_replan_chunk(self) -> bool:
        if len(self._action_queue) == 0:
            return True
        if len(self._action_queue) <= int(getattr(self.config, "system1_min_queue_size", 0)):
            return True
        replan_every = int(getattr(self.config, "system1_replan_every_n_steps", 0))
        if replan_every <= 0:
            return False
        if self._last_replan_step is None:
            return True
        return (self._inference_step - self._last_replan_step) >= replan_every

    def _get_backbone_outputs_for_inference(
        self, groot_inputs: dict[str, Tensor], *, force_refresh: bool
    ) -> BatchFeature:
        refresh_interval = self._get_system2_interval_steps()
        should_refresh = self._should_refresh_cache(
            current_step=self._inference_step,
            last_refresh_step=self._last_backbone_refresh_inference_step,
            refresh_interval=refresh_interval,
            force_refresh=force_refresh,
            has_cache=self._cached_backbone_outputs is not None,
        )
        if should_refresh:
            backbone_outputs = self._groot_model.run_backbone(groot_inputs)
            self._cached_backbone_outputs = self._detach_batch_feature(backbone_outputs)
            self._last_backbone_refresh_inference_step = self._inference_step
            self._system2_refresh_count_inference += 1
        return self._cached_backbone_outputs  # type: ignore[return-value]

    def _get_backbone_outputs_for_training(
        self, groot_inputs: dict[str, Tensor], *, force_refresh: bool
    ) -> BatchFeature:
        refresh_interval = self._get_system2_interval_steps()
        should_refresh = self._should_refresh_cache(
            current_step=self._train_forward_step,
            last_refresh_step=self._last_backbone_refresh_train_step,
            refresh_interval=refresh_interval,
            force_refresh=force_refresh,
            has_cache=self._cached_train_backbone_outputs is not None,
        )
        if should_refresh:
            backbone_outputs = self._groot_model.run_backbone(groot_inputs)
            # Keep gradient path for the current forward, but store detached tensors for future stale-use steps.
            self._cached_train_backbone_outputs = self._detach_batch_feature(backbone_outputs)
            self._last_backbone_refresh_train_step = self._train_forward_step
            self._system2_refresh_count_train += 1
            return backbone_outputs
        return self._clone_batch_feature(self._cached_train_backbone_outputs)  # type: ignore[arg-type]

    _system1_visual_source_warned: bool = False

    def _get_fresh_visual_features(self, groot_inputs: dict[str, Tensor]) -> Tensor | None:
        """Run ViT-only fast path if backbone supports it. Returns projected visual tokens or None."""
        if not hasattr(self._groot_model, "run_visual_only"):
            return None
        from lerobot.policies.grootCoT.groot_n1 import QwenBackbone
        if not isinstance(self._groot_model.backbone, QwenBackbone):
            return None
        source = getattr(self.config, "system1_visual_source", "vit")
        if source != "vit":
            if not self._system1_visual_source_warned:
                print(
                    f"[GROOT] system1_visual_source='{source}' is not yet implemented; "
                    f"falling back to 'vit' (pure ViT output)."
                )
                self._system1_visual_source_warned = True
        visual_output = self._groot_model.run_visual_only(groot_inputs)
        return visual_output.get("visual_features")

    def _predict_action_chunk_from_inputs(
        self,
        *,
        groot_inputs: dict[str, Tensor],
        force_backbone_refresh: bool,
        use_cached_dual_rate: bool = True,
    ) -> Tensor:
        device = next(self.parameters()).device
        use_cfg = (
            bool(getattr(self.config, "recap_enable", False))
            and bool(getattr(self.config, "recap_adv_indicator_use_cfg", False))
            and float(getattr(self.config, "recap_cfg_scale", 1.0)) > 1.0
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            # System 1 visual features: always fresh from current camera frame
            fresh_visual = self._get_fresh_visual_features(groot_inputs)

            if not use_cfg:
                if self._dual_rate_enabled() and use_cached_dual_rate:
                    backbone_outputs = self._get_backbone_outputs_for_inference(
                        groot_inputs,
                        force_refresh=force_backbone_refresh,
                    )
                    # Action head mutates backbone_output in-place; preserve cache by passing a clone.
                    backbone_outputs = self._clone_batch_feature(backbone_outputs)
                    outputs = self._groot_model.run_action_head(
                        inputs=groot_inputs,
                        backbone_outputs=backbone_outputs,
                        is_training=False,
                        fresh_visual_features=fresh_visual,
                    )
                elif self._dual_rate_enabled():
                    backbone_outputs = self._groot_model.run_backbone(groot_inputs)
                    outputs = self._groot_model.run_action_head(
                        inputs=groot_inputs,
                        backbone_outputs=backbone_outputs,
                        is_training=False,
                        fresh_visual_features=fresh_visual,
                    )
                else:
                    outputs = self._groot_model.get_action(groot_inputs)
            else:
                cond_inputs = dict(groot_inputs)
                uncond_inputs = dict(groot_inputs)
                cond_inputs = self._set_indicator_tensor(
                    cond_inputs,
                    float(getattr(self.config, "recap_adv_indicator_cond_value", 1.0)),
                )
                uncond_inputs = self._set_indicator_tensor(
                    uncond_inputs,
                    float(getattr(self.config, "recap_adv_indicator_null_value", -1.0)),
                )
                if self._dual_rate_enabled() and use_cached_dual_rate:
                    backbone_outputs = self._get_backbone_outputs_for_inference(
                        cond_inputs,
                        force_refresh=force_backbone_refresh,
                    )
                elif self._dual_rate_enabled():
                    backbone_outputs = self._groot_model.run_backbone(cond_inputs)
                else:
                    backbone_outputs = self._groot_model.run_backbone(cond_inputs)

                cond_outputs = self._groot_model.run_action_head(
                    inputs=cond_inputs,
                    backbone_outputs=self._clone_batch_feature(backbone_outputs),
                    is_training=False,
                    fresh_visual_features=fresh_visual,
                )
                uncond_outputs = self._groot_model.run_action_head(
                    inputs=uncond_inputs,
                    backbone_outputs=self._clone_batch_feature(backbone_outputs),
                    is_training=False,
                    fresh_visual_features=fresh_visual,
                )
                actions = apply_cfg_guidance_velocity(
                    cond_outputs.get("action_pred"),
                    uncond_outputs.get("action_pred"),
                    float(getattr(self.config, "recap_cfg_scale", 1.0)),
                )

        if not use_cfg:
            actions = outputs.get("action_pred")
        original_action_dim = self.config.output_features["action"].shape[0]
        return actions[:, :, :original_action_dim]

    def _compute_action_chunk_sync(
        self,
        *,
        groot_inputs: dict[str, Tensor],
        force_backbone_refresh: bool,
        use_cached_dual_rate: bool = True,
    ) -> Tensor:
        return self._predict_action_chunk_from_inputs(
            groot_inputs=groot_inputs,
            force_backbone_refresh=force_backbone_refresh,
            use_cached_dual_rate=use_cached_dual_rate,
        )

    def get_dual_rate_runtime_stats(self) -> dict[str, float | int | bool]:
        now_perf = time.perf_counter()
        elapsed_s = max(1e-6, now_perf - self._inference_start_time)
        s1_hz = float(self._inference_step) / elapsed_s

        with self._async_lock:
            worker = self._async_thread
            worker_alive = bool(worker is not None and worker.is_alive())
            worker_busy = bool(self._async_worker_busy)
            chunks_ready = int(len(self._async_ready_chunks))
            async_refresh_count = int(self._async_refresh_count)
            async_last_refresh_time = self._async_last_refresh_time

        if async_last_refresh_time is None:
            s2_last_refresh_age_ms = float("inf")
        else:
            s2_last_refresh_age_ms = (time.time() - async_last_refresh_time) * 1000.0

        return {
            "s1_inference_hz_est": s1_hz,
            "s1_fallback_count": int(self._s1_fallback_count),
            "s1_queue_len": int(len(self._action_queue)),
            "s2_worker_alive": worker_alive,
            "s2_worker_busy": worker_busy,
            "s2_chunks_ready": chunks_ready,
            "s2_last_refresh_age_ms": s2_last_refresh_age_ms,
            "s2_refresh_count": async_refresh_count,
        }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        batch = self._prepare_recap_indicator_batch(batch, apply_dropout=True)

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        groot_inputs = self._build_groot_inputs(batch, include_action=True)

        # Get device from model parameters
        device = next(self.parameters()).device

        # Run GR00T forward under bf16 autocast when enabled to reduce activation memory
        # Rationale: Matches original GR00T finetuning (bf16 compute, fp32 params) and avoids fp32 upcasts.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            if self._dual_rate_train_enabled():
                force_refresh = self._train_forward_step == 0
                backbone_outputs = self._get_backbone_outputs_for_training(
                    groot_inputs,
                    force_refresh=force_refresh,
                )
                # Extract visual_features captured by ViT hook during backbone forward
                fresh_visual = backbone_outputs.get("visual_features")
                # Visual feature dropout: randomly zero fresh features to teach the
                # DiT to function with cached System 2 features alone.
                if fresh_visual is not None and self.training:
                    dropout_p = getattr(self.config, "visual_dropout_p", 0.2)
                    if dropout_p > 0 and torch.rand(1).item() < dropout_p:
                        fresh_visual = torch.zeros_like(fresh_visual)
                outputs = self._groot_model.run_action_head(
                    inputs=groot_inputs,
                    backbone_outputs=backbone_outputs,
                    is_training=True,
                    fresh_visual_features=fresh_visual,
                )
            else:
                _projector_only = getattr(self.config, "train_vlm_projector_only", False)
                if _projector_only:
                    # Stage 1: Qwen is fully frozen. Run under no_grad to skip
                    # activation storage for 36 LLM layers (~15-20GB saved).
                    with torch.no_grad():
                        backbone_outputs = self._groot_model.run_backbone(groot_inputs)
                    # Detach backbone features so gradients don't flow into Qwen.
                    for k in list(backbone_outputs.keys()):
                        if isinstance(backbone_outputs[k], torch.Tensor) and backbone_outputs[k].requires_grad:
                            backbone_outputs[k] = backbone_outputs[k].detach().requires_grad_(True)
                    outputs = self._groot_model.run_action_head(
                        inputs=groot_inputs,
                        backbone_outputs=backbone_outputs,
                        is_training=True,
                    )
                else:
                    # Run backbone under no_grad -- ViT and LLM are frozen or
                    # nearly frozen, and projector gradients come from the action
                    # head's cross-attention backward. Detach output so gradient
                    # flows through projector → action head only.
                    with torch.no_grad():
                        backbone_outputs = self._groot_model.run_backbone(groot_inputs)
                    for k in list(backbone_outputs.keys()):
                        if isinstance(backbone_outputs[k], torch.Tensor) and backbone_outputs[k].requires_grad:
                            backbone_outputs[k] = backbone_outputs[k].detach().requires_grad_(True)
                    outputs = self._groot_model.run_action_head(
                        inputs=groot_inputs,
                        backbone_outputs=backbone_outputs,
                        is_training=True,
                    )

        self._train_forward_step += 1

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}
        for key in (
            "loss_primary",
            "loss_secondary",
            "loss_primary_weighted",
            "loss_secondary_weighted",
            "loss_upper",
            "loss_lower",
            "loss_upper_weighted",
            "loss_lower_weighted",
        ):
            value = outputs.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                loss_dict[key] = value.detach().float().cpu().item()

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim).
        """
        self.eval()
        batch = self._prepare_recap_indicator_batch(batch, apply_dropout=False)

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        groot_inputs = self._build_groot_inputs(batch, include_action=False)
        actions = self._compute_action_chunk_sync(
            groot_inputs=groot_inputs,
            force_backbone_refresh=False,
        )
        self._last_replan_step = self._inference_step
        self._inference_step += 1
        return actions

    def get_subtask_overrides(self) -> dict[str, Any]:
        """Return current sub-task state for injection into the processor pipeline.

        The inference loop should merge these into the transition's complementary
        data **before** the processor runs, so the language and bbox reach
        tokenization and grounded reference frame rendering.
        """
        overrides: dict[str, Any] = {}
        if self._current_subtask_text is not None:
            overrides["current_subtask_text"] = self._current_subtask_text
        if self._current_target_bbox is not None:
            overrides["target_bbox"] = self._current_target_bbox
        return overrides

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()
        batch = self._prepare_recap_indicator_batch(batch, apply_dropout=False)
        if not self._async_dual_rate_enabled():
            if self._should_replan_chunk():
                groot_inputs = self._build_groot_inputs(batch, include_action=False)
                force_refresh = (
                    bool(getattr(self.config, "dual_rate_force_backbone_refresh_on_reset", True))
                    and self._last_backbone_refresh_inference_step is None
                )
                actions = self._compute_action_chunk_sync(
                    groot_inputs=groot_inputs,
                    force_backbone_refresh=force_refresh,
                )
                self._action_queue.clear()
                self._action_queue.extend(actions.transpose(0, 1))
                self._last_replan_step = self._inference_step
            if len(self._action_queue) == 0:
                raise RuntimeError("Action queue is empty after replanning.")
            action = self._action_queue.popleft()
            self._last_emitted_action = action.detach().clone()
            self._inference_step += 1
            return action

        # Async dual-rate path: never block System-1 action emission.
        self._start_async_worker_if_needed()
        groot_inputs = self._build_groot_inputs(batch, include_action=False)
        self._publish_latest_inputs_for_worker(groot_inputs)

        # Optional one-time warmup (disabled by default to avoid blocking).
        if (
            bool(getattr(self.config, "system2_async_startup_warmup", False))
            and not self._async_startup_warmup_done
            and len(self._action_queue) == 0
        ):
            deadline = time.monotonic() + 2.0 * self._get_system2_async_period_s()
            while time.monotonic() < deadline and len(self._action_queue) == 0:
                ready_chunk = self._pop_latest_async_chunk()
                if ready_chunk is not None:
                    self._action_queue.clear()
                    self._action_queue.extend(ready_chunk)
                    self._last_replan_step = self._inference_step
                    break
                time.sleep(0.001)
            self._async_startup_warmup_done = True

        ready_chunk = self._pop_latest_async_chunk()
        if ready_chunk is not None:
            self._action_queue.clear()
            self._action_queue.extend(ready_chunk)
            self._last_replan_step = self._inference_step

        if len(self._action_queue) > 0:
            action = self._action_queue.popleft()
        elif self._cached_backbone_outputs is not None:
            # System 1 fast-visual path: generate actions using cached System 2
            # backbone features + fresh ViT features from the current camera frame.
            # This fills the gap between System 2 updates with reactive actions.
            device = next(self.parameters()).device
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
                fresh_visual = self._get_fresh_visual_features(groot_inputs)
                backbone_clone = self._clone_batch_feature(self._cached_backbone_outputs)
                outputs = self._groot_model.run_action_head(
                    inputs=groot_inputs,
                    backbone_outputs=backbone_clone,
                    is_training=False,
                    fresh_visual_features=fresh_visual,
                )
            actions = outputs.get("action_pred")
            original_action_dim = self.config.output_features["action"].shape[0]
            actions = actions[:, :, :original_action_dim]
            action_chunk = actions.transpose(0, 1).detach().clone()
            self._action_queue.clear()
            self._action_queue.extend(action_chunk)
            self._last_replan_step = self._inference_step
            action = self._action_queue.popleft()
        else:
            action = self._state_based_fallback_action(batch)
            self._last_fallback_action = action.detach().clone()
            self._s1_fallback_count += 1

        self._last_emitted_action = action.detach().clone()
        self._maybe_log_async_error()
        self._inference_step += 1
        return action

    def forward_value(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Forward value head on System-2 (Qwen) features."""
        batch = self._prepare_recap_indicator_batch(batch, apply_dropout=False)
        groot_inputs = self._build_groot_inputs(batch, include_action=False)
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            value_outputs = self._groot_model.predict_value(groot_inputs)
        return value_outputs["value_logits"], value_outputs["value_scalar"]

    @torch.no_grad()
    def predict_value(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        self.eval()
        return self.forward_value(batch)

    @torch.no_grad()
    def extract_cot_trace(
        self,
        batch: dict[str, Tensor],
        *,
        cot_session: "Any | None" = None,
        dataset_meta: "dict | None" = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict[str, str | int]]:
        """Generate reasoning traces and update System 1 sub-task state."""
        self.eval()
        groot_inputs = self._build_groot_inputs(batch, include_action=False)
        traces = self._groot_model.extract_cot_trace(
            groot_inputs,
            cot_session=cot_session,
            dataset_meta=dataset_meta,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        self._update_subtask_from_traces(traces)
        return traces

    def _update_subtask_from_traces(self, traces: list[dict]) -> None:
        """Extract sub-task text and bbox from CoT traces into persistent state."""
        if not traces or not traces[0].get("parse_ok"):
            return
        from lerobot.policies.grootCoT.cot_schema import extract_subtask_fields
        parsed = traces[0].get("parsed_json")
        if not isinstance(parsed, dict):
            return
        subtask_text, target_bbox = extract_subtask_fields(parsed)
        if subtask_text:
            self._current_subtask_text = subtask_text
        if target_bbox is not None:
            self._current_target_bbox = target_bbox

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
