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
from collections import deque
from typing import Any

import torch
from torch import Tensor
from transformers import BatchFeature

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

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlm_projector=tune_vlm_projector,
            tune_vlln=tune_vlln,
            tune_top_llm_layers=tune_top_llm_layers,
            model_id=self.config.vlm_processor_model_id,
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
            # Optional weighted upper/lower body loss split
            upper_body_joint_indices=self.config.upper_body_joint_indices,
            lower_body_joint_indices=self.config.lower_body_joint_indices,
            upper_body_loss_weight=self.config.upper_body_loss_weight,
            lower_body_loss_weight=self.config.lower_body_loss_weight,
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

        if extra_dims:
            print(f"[GROOT] Extra observation projections enabled for keys: {sorted(extra_dims.keys())}")
        return extra_dims

    def reset(self):
        """Reset policy state when environment resets."""
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

    def get_optim_params(self) -> dict:
        return self.parameters()

    @staticmethod
    def _build_groot_inputs(batch: dict[str, Tensor], include_action: bool) -> dict[str, Tensor]:
        """Filter a preprocessed batch to keys consumed by GR00T."""
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

    def _dual_rate_enabled(self) -> bool:
        return bool(getattr(self.config, "dual_rate_enable", False))

    def _dual_rate_train_enabled(self) -> bool:
        return self._dual_rate_enabled() and bool(getattr(self.config, "dual_rate_apply_in_train", False))

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

    def _predict_action_chunk_from_inputs(
        self,
        *,
        groot_inputs: dict[str, Tensor],
        force_backbone_refresh: bool,
    ) -> Tensor:
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            if self._dual_rate_enabled():
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
                )
            else:
                outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")
        original_action_dim = self.config.output_features["action"].shape[0]
        return actions[:, :, :original_action_dim]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
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
                outputs = self._groot_model.run_action_head(
                    inputs=groot_inputs,
                    backbone_outputs=backbone_outputs,
                    is_training=True,
                )
            else:
                outputs = self._groot_model.forward(groot_inputs)

        self._train_forward_step += 1

        # Isaac-GR00T returns a BatchFeature; loss key is typically 'loss'
        loss = outputs.get("loss")

        loss_dict = {"loss": loss.item()}
        for key in ("loss_upper", "loss_lower", "loss_upper_weighted", "loss_lower_weighted"):
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

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        groot_inputs = self._build_groot_inputs(batch, include_action=False)
        actions = self._predict_action_chunk_from_inputs(
            groot_inputs=groot_inputs,
            force_backbone_refresh=False,
        )
        self._last_replan_step = self._inference_step
        self._inference_step += 1
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if self._should_replan_chunk():
            groot_inputs = self._build_groot_inputs(batch, include_action=False)
            force_refresh = (
                bool(getattr(self.config, "dual_rate_force_backbone_refresh_on_reset", True))
                and self._last_backbone_refresh_inference_step is None
            )
            actions = self._predict_action_chunk_from_inputs(
                groot_inputs=groot_inputs,
                force_backbone_refresh=force_refresh,
            )
            self._action_queue.clear()
            self._action_queue.extend(actions.transpose(0, 1))
            self._last_replan_step = self._inference_step
        if len(self._action_queue) == 0:
            raise RuntimeError("Action queue is empty after replanning.")
        action = self._action_queue.popleft()
        self._inference_step += 1
        return action

    @torch.no_grad()
    def extract_cot_trace(
        self,
        batch: dict[str, Tensor],
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict[str, str | int]]:
        """Generate and return Qwen reasoning traces for the current batch."""
        self.eval()
        groot_inputs = self._build_groot_inputs(batch, include_action=False)
        return self._groot_model.extract_cot_trace(
            groot_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

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
