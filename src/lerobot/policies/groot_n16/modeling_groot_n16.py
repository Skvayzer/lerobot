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
GR00T N1.6 Policy Wrapper for LeRobot Integration

Wraps the Gr00tN1d6 model from the gr00t package (installed at /home/cosmos/Isaac-GR00T).
Key differences from N1.5:
- 32-layer AlternateVLDiT diffusion head
- Eagle-Block2A-2B-v2 backbone (same family)
- Input keys: pixel_values, input_ids, attention_mask (no eagle_ prefix)
- max_state_dim=29, max_action_dim=29
- ROCm compatibility: patching eager attention after model load
"""

import builtins
import os
from collections import deque
from pathlib import Path
from typing import TypeVar

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot_n16.configuration_groot_n16 import GrootN16Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

T = TypeVar("T", bound="GrootN16Policy")


def _force_eager_attn(cfg, depth: int = 0) -> None:
    """Recursively patch all sub-configs to use eager attention instead of flash_attention_2."""
    if cfg is None or depth > 6:
        return
    for attr in ("_attn_implementation", "_attn_implementation_internal"):
        if getattr(cfg, attr, None) == "flash_attention_2":
            setattr(cfg, attr, "eager")
    for name in list(vars(cfg)):
        if name.endswith("_config"):
            _force_eager_attn(getattr(cfg, name, None), depth + 1)


def _load_gr00t_n1d6(base_model_path: str, use_flash_attention: bool) -> "Gr00tN1d6":
    """Load Gr00tN1d6 model with ROCm-compatible attention.

    ROCm strategy: override the pretrained model's use_flash_attention=True
    to False in the internal config before loading. The patched eagle_backbone.py
    then sets eager attention on the Eagle config before AutoModel.from_config().
    """
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6

    if use_flash_attention:
        return Gr00tN1d6.from_pretrained(base_model_path, trust_remote_code=True)

    # ROCm path: load internal config and override use_flash_attention to False.
    # gr00t_n1d6.py passes config.use_flash_attention to EagleBackbone.__init__,
    # which (after our eagle_backbone.py patch) sets eager attention on the Eagle config.
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config as _Gr00tInternalCfg
    internal_cfg = _Gr00tInternalCfg.from_pretrained(base_model_path)
    internal_cfg.use_flash_attention = False

    model = Gr00tN1d6.from_pretrained(
        base_model_path,
        config=internal_cfg,
        trust_remote_code=True,
    )
    # Belt-and-suspenders: ensure eager attention on backbone config
    _force_eager_attn(model.backbone.model.config)
    return model


class GrootN16Policy(PreTrainedPolicy):
    """Wrapper around GR00T N1.6 (Gr00tN1d6) for LeRobot integration."""

    name = "groot_n16"
    config_class = GrootN16Config

    def __init__(self, config: GrootN16Config, **kwargs):
        super().__init__(config, **kwargs)
        config.validate_features()
        self.config = config

        self._model = self._create_model()
        self.reset()

    def _create_model(self):
        """Load Gr00tN1d6 from HuggingFace and configure trainable params."""
        print(
            f"[GROOT-N16] Loading GR00T N1.6 from: {self.config.base_model_path}\n"
            f"  tune_llm={self.config.tune_llm}, tune_visual={self.config.tune_visual}, "
            f"tune_projector={self.config.tune_projector}, "
            f"tune_diffusion_model={self.config.tune_diffusion_model}"
        )

        model = _load_gr00t_n1d6(
            base_model_path=self.config.base_model_path,
            use_flash_attention=self.config.use_flash_attention,
        )

        if self.config.use_bf16:
            model = model.to(torch.bfloat16)
            model.config.model_dtype = "bfloat16"

        self._apply_freeze_config(model)
        return model

    def _apply_freeze_config(self, model) -> None:
        """Freeze/unfreeze model components based on tune_* flags."""
        # Freeze all
        for p in model.parameters():
            p.requires_grad_(False)

        # Unfreeze action head diffusion model
        if self.config.tune_diffusion_model:
            for p in model.action_head.model.parameters():
                p.requires_grad_(True)

        # Unfreeze action head projector components (state/action encoders/decoders)
        if self.config.tune_projector:
            for p in model.action_head.state_encoder.parameters():
                p.requires_grad_(True)
            for p in model.action_head.action_encoder.parameters():
                p.requires_grad_(True)
            for p in model.action_head.action_decoder.parameters():
                p.requires_grad_(True)
            if hasattr(model.action_head, "position_embedding"):
                for p in model.action_head.position_embedding.parameters():
                    p.requires_grad_(True)
            if hasattr(model.action_head, "vlln"):
                for p in model.action_head.vlln.parameters():
                    p.requires_grad_(True)
            if model.action_head.mask_token is not None:
                model.action_head.mask_token.requires_grad_(True)

        # Unfreeze backbone visual encoder
        if self.config.tune_visual:
            for p in model.backbone.model.vision_model.parameters():
                p.requires_grad_(True)
            for p in model.backbone.model.mlp1.parameters():
                p.requires_grad_(True)

        # Unfreeze backbone LLM
        if self.config.tune_llm:
            for p in model.backbone.model.language_model.parameters():
                p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(
            f"[GROOT-N16] Trainable params: {n_trainable:,} / {n_total:,} "
            f"({100 * n_trainable / max(n_total, 1):.2f}%)"
        )

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: GrootN16Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load policy from pretrained base model or fine-tuned checkpoint."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from huggingface_hub.errors import HfHubHTTPError

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False

        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(os.path.join(model_id, SAFETENSORS_SINGLE_FILE))
            else:
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            print(f"[GROOT-N16] Loading fine-tuned checkpoint from: {pretrained_name_or_path}")
            return super().from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                config=config,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            )

        print(f"[GROOT-N16] Loading base GR00T N1.6 model from: {pretrained_name_or_path}")
        if config is None:
            config = GrootN16Config(base_model_path=str(pretrained_name_or_path))
            if not config.input_features:
                config.input_features = {
                    f"{OBS_IMAGES}.camera": PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, 224, 224),
                    ),
                }
        else:
            config.base_model_path = str(pretrained_name_or_path)

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        policy = cls(config)
        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward pass."""
        # N1.6 uses pixel_values/input_ids/attention_mask (no eagle_ prefix)
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        n16_inputs = {
            k: v
            for k, v in batch.items()
            if (
                k in allowed_base
                or k in ("pixel_values", "input_ids", "attention_mask", "image_grid_thw")
            )
            and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._model.forward(n16_inputs)

        loss = outputs.get("loss")
        loss_dict = {"loss": loss.item()}
        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions (inference only)."""
        self.eval()

        allowed_base = {"state", "state_mask", "embodiment_id"}
        n16_inputs = {
            k: v
            for k, v in batch.items()
            if (
                k in allowed_base
                or k in ("pixel_values", "input_ids", "attention_mask", "image_grid_thw")
            )
            and not (k.startswith("next.") or k == "info")
        }

        device = next(self.parameters()).device

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._model.get_action(n16_inputs)

        actions = outputs.get("action_pred")  # (B, action_horizon, max_action_dim)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()
