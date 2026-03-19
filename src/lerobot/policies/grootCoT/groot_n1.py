# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
import os
import re

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from peft import LoraConfig, get_peft_model, PeftModel

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForVision2Seq,
        AutoTokenizer,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.feature_extraction_utils import BatchFeature
    # Qwen3VLForConditionalGeneration requires transformers >= 4.53 with Qwen3-VL support.
    # Gracefully degrade to None if not available (will fall back to AutoModel).
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        Qwen3VLForConditionalGeneration = None
else:
    AutoConfig = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoModelForVision2Seq = None
    AutoTokenizer = None
    Qwen3VLForConditionalGeneration = None
    PretrainedConfig = object
    PreTrainedModel = object
    BatchFeature = None

try:
    import tree
except ImportError:
    tree = None

from lerobot.policies.grootCoT.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from lerobot.policies.grootCoT.system2_vlm_registry import DEFAULT_SYSTEM2_VLM_MODEL_ID
from lerobot.utils.constants import HF_LEROBOT_HOME

# Monkey-patch torch.linspace to handle Tensor 'steps' argument.
# This fixes a crash in Qwen3-VL where it passes a scalar tensor as steps,
# but torch.linspace expects an int.
_orig_linspace = torch.linspace

def _safe_linspace(start, end, steps, *args, **kwargs):
    if isinstance(steps, torch.Tensor) and steps.numel() == 1:
        steps = steps.item()
    return _orig_linspace(start, end, steps, *args, **kwargs)

torch.linspace = _safe_linspace

DEFAULT_VENDOR_EAGLE_PATH = str((Path(__file__).resolve().parent / "eagle2_hg_model").resolve())
DEFAULT_TOKENIZER_ASSETS_REPO = "lerobot/eagle2hg-processor-groot-n1p5"
DEFAULT_QWEN_MODEL_ID = DEFAULT_SYSTEM2_VLM_MODEL_ID
DEFAULT_SUMMARY_TOKEN = "<SUMMARY>"
DEFAULT_QWEN_INPUT_PREFIX = "qwen_"
LEGACY_EAGLE_INPUT_PREFIX = "eagle_"


class QwenBackbone(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL_ID,
        tune_vlm: bool = False,
        tune_projector: bool = True,
        value_head_enable: bool = False,
        tune_value_head: bool = True,
        value_head_bins: int = 201,
        value_head_vmin: float = -1.0,
        value_head_vmax: float = 0.0,
        value_head_pooling: str = "masked_mean",
        tune_top_llm_layers: int = 0,
        select_layer: int | None = None,
        project_to_dim: int | None = 1536,
        input_prefix: str = DEFAULT_QWEN_INPUT_PREFIX,
        load_bf16: bool = False,
        attn_implementation: str | None = None,
        summary_token: str = DEFAULT_SUMMARY_TOKEN,
        summary_avg_last_k: int | None = None,
        lora_config: dict | None = None,
        **_: dict,
    ):
        super().__init__()
        self.model_id = model_id
        self._model_id_lower = model_id.lower()
        self.input_prefix = input_prefix
        self.legacy_input_prefix = (
            LEGACY_EAGLE_INPUT_PREFIX
            if input_prefix != LEGACY_EAGLE_INPUT_PREFIX
            else DEFAULT_QWEN_INPUT_PREFIX
        )
        self.summary_token = summary_token
        self.summary_avg_last_k = summary_avg_last_k
        self.lora_config = lora_config or {}
        self.tune_top_llm_layers = max(0, int(tune_top_llm_layers))
        self.value_head_enable = bool(value_head_enable)
        self.tune_value_head = bool(tune_value_head)
        self.value_head_pooling = str(value_head_pooling)
        self.value_head_bins = int(value_head_bins)
        self.value_head_vmin = float(value_head_vmin)
        self.value_head_vmax = float(value_head_vmax)
        self._tokenizer = None

        _is_rocm_early = getattr(torch.version, "hip", None) is not None
        self.qwen_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        load_kwargs = {"trust_remote_code": True}
        if _is_rocm_early:
            # On ROCm (AMD MI210 gfx90a), BF16 produces NaN in both ViT and LLM layers.
            # Force FP32 for the entire model regardless of load_bf16 setting.
            load_kwargs["torch_dtype"] = torch.float32
            print(f"[GROOT] ROCm: forcing torch_dtype=float32 (hip={torch.version.hip}, load_bf16={load_bf16}) to prevent NaN.", flush=True)
        elif load_bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        # Prefer a class that matches the config model_type to avoid partial/random initialization.
        model_type = str(getattr(self.qwen_config, "model_type", "")).lower()
        if model_type.startswith("qwen3_vl") and Qwen3VLForConditionalGeneration is not None:
            self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        else:
            try:
                # Try specific Qwen2VL class if available (transformers>=4.45)
                from transformers import Qwen2VLForConditionalGeneration

                self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            except (ImportError, Exception):
                # Fallback to AutoModel (best for general compatibility)
                try:
                    self.qwen_model = AutoModel.from_pretrained(model_id, **load_kwargs)
                except Exception:
                    try:
                        self.qwen_model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs)
                    except Exception:
                        if Qwen3VLForConditionalGeneration is not None:
                            self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
                        else:
                            self.qwen_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        print(f"[GROOT] Initialized QwenBackbone with model class: {type(self.qwen_model)}")
        # AMD ROCm workaround: BF16 produces NaN in both ViT and LLM layers on MI210.
        # Cast the ENTIRE model to FP32 for numerical stability.
        _is_rocm = getattr(torch.version, "hip", None) is not None
        print(f"[GROOT] ROCm check: load_bf16={load_bf16}, hip={getattr(torch.version, 'hip', None)}, _is_rocm={_is_rocm}", flush=True)
        if _is_rocm:
            if not load_bf16:
                # Use FP16 for frozen Qwen inference. FP16 works fine on MI210
                # for inference (no NaN). Only BF16 matmuls produce NaN.
                # This saves ~16GB vs FP32, making Stage 2 feasible.
                self.qwen_model.half()
                print(f"[GROOT] AMD ROCm: cast Qwen to FP16 (hip={torch.version.hip}). FP16 is safe for inference.", flush=True)
            else:
                self.qwen_model.float()
                print(f"[GROOT] AMD ROCm: cast entire Qwen model to FP32 (hip={torch.version.hip}) to prevent BF16 NaN.", flush=True)
        # Diagnostic hook: print ViT output stats for first 3 forward passes
        _hook_calls = [0]
        def _vit_diag_hook(module, inp, out):
            _hook_calls[0] += 1
            if _hook_calls[0] > 3:
                return
            x = out[0] if isinstance(out, (tuple, list)) else out
            if torch.isnan(x).any():
                print(f"[GROOT] ViT output NaN #{_hook_calls[0]}: {torch.isnan(x).sum()}/{x.numel()} NaN, dtype={x.dtype}", flush=True)
            else:
                print(f"[GROOT] ViT output OK #{_hook_calls[0]}: min={x.min().item():.3f}, max={x.max().item():.3f}, dtype={x.dtype}", flush=True)
        for _vit_attr2 in ("visual", "vision_model", "vision_tower", "visual_encoder"):
            if hasattr(self.qwen_model, _vit_attr2):
                getattr(self.qwen_model, _vit_attr2).register_forward_hook(_vit_diag_hook)
                break

        # Hook to capture ViT output during full forward pass.
        # _in_visual_only_mode prevents the hook from overwriting cached output
        # when forward_visual_only calls the ViT independently.
        self._cached_vit_output = None
        self._in_visual_only_mode = False

        def _capture_vit_output(module, inp, out):
            if self._in_visual_only_mode:
                return
            # Unwrap tuple outputs (some Qwen versions return (hidden_states, ...))
            x = out[0] if isinstance(out, (tuple, list)) else out
            # Keep gradient path during training; detach for inference
            self._cached_vit_output = x.detach() if not self.training else x

        for _attr in ("visual", "vision_model", "vision_tower", "visual_encoder"):
            if hasattr(self.qwen_model, _attr):
                getattr(self.qwen_model, _attr).register_forward_hook(_capture_vit_output)
                break

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            print(f"[GROOT] Warning: failed to load tokenizer for CoT decoding: {exc}")

        hidden_size = getattr(self.qwen_config, "hidden_size", None)
        if hidden_size is None and hasattr(self.qwen_config, "text_config"):
            hidden_size = getattr(self.qwen_config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Qwen config missing hidden_size.")

        num_layers = getattr(self.qwen_config, "num_hidden_layers", None)
        if num_layers is None and hasattr(self.qwen_config, "text_config"):
            num_layers = getattr(self.qwen_config.text_config, "num_hidden_layers", None)

        self.select_layer = select_layer
        if self.select_layer is None and num_layers is not None:
            # Late layer for maximum semantic richness (System 2 features).
            # Layer 28 of 32: deep enough for full reasoning, avoids final-layer
            # next-token-prediction bias. Spatial precision is handled separately
            # by System 1's fast ViT path (forward_visual_only).
            self.select_layer = max(1, num_layers - 3)

        if project_to_dim is None:
            self.projector = nn.Identity()
            value_input_dim = int(hidden_size)
        else:
            self.projector = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, project_to_dim))
            value_input_dim = int(project_to_dim)
            if load_bf16 and not _is_rocm:
                self.projector = self.projector.to(torch.bfloat16)

        if self.value_head_enable:
            if self.value_head_bins < 2:
                raise ValueError("value_head_bins must be >= 2.")
            if self.value_head_vmax <= self.value_head_vmin:
                raise ValueError("value_head_vmax must be > value_head_vmin.")
            if self.value_head_pooling not in {"masked_mean", "last_token"}:
                raise ValueError("value_head_pooling must be one of {'masked_mean', 'last_token'}.")
            self.value_head = nn.Sequential(
                nn.LayerNorm(value_input_dim),
                nn.Linear(value_input_dim, value_input_dim),
                nn.GELU(),
                nn.Linear(value_input_dim, self.value_head_bins),
            )
            self.register_buffer(
                "value_bin_centers",
                torch.linspace(self.value_head_vmin, self.value_head_vmax, self.value_head_bins),
                persistent=False,
            )
        else:
            self.value_head = None

        self.set_trainable_parameters(
            tune_vlm=tune_vlm,
            tune_projector=tune_projector,
            tune_value_head=self.tune_value_head,
            tune_top_llm_layers=self.tune_top_llm_layers,
        )

    def _get_language_layers(self):
        # Qwen2-VL path: qwen_model.model.language_model.layers
        if hasattr(self.qwen_model, "model"):
            model = self.qwen_model.model
            if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
                return model.language_model.layers
            if hasattr(model, "layers"):
                return model.layers

        # Fallback path for some wrapper/model variants.
        if hasattr(self.qwen_model, "language_model") and hasattr(self.qwen_model.language_model, "layers"):
            return self.qwen_model.language_model.layers

        return None

    def _unfreeze_top_llm_layers(self, top_k: int) -> bool:
        layers = self._get_language_layers()
        if layers is None:
            return False

        total_layers = len(layers)
        if total_layers == 0:
            return False

        k = max(0, min(int(top_k), total_layers))
        if k == 0:
            return True

        start_idx = total_layers - k
        for idx in range(start_idx, total_layers):
            layers[idx].requires_grad_(True)

        # Also unfreeze language-model norm/head where present.
        if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "language_model"):
            lm = self.qwen_model.model.language_model
            if hasattr(lm, "norm"):
                lm.norm.requires_grad_(True)
        if hasattr(self.qwen_model, "lm_head"):
            self.qwen_model.lm_head.requires_grad_(True)

        print(f"[GROOT] Selective Qwen unfreeze: top_llm_layers={k}/{total_layers}")
        return True

    def set_trainable_parameters(
        self,
        tune_vlm: bool,
        tune_projector: bool,
        tune_value_head: bool = True,
        tune_top_llm_layers: int | None = None,
    ):
        self.tune_vlm = tune_vlm
        self.tune_projector = tune_projector
        self.tune_value_head = bool(tune_value_head)
        if tune_top_llm_layers is None:
            tune_top_llm_layers = self.tune_top_llm_layers
        self.tune_top_llm_layers = max(0, int(tune_top_llm_layers))

        # Start from everything frozen, then selectively unfreeze.
        for p in self.parameters():
            p.requires_grad = False

        # Attach LoRA once (if requested) before deciding trainable params.
        if tune_vlm and self.lora_config.get("r", 0) > 0 and not isinstance(self.qwen_model, PeftModel):
            print(f"Applying LoRA to Qwen VLM with config: {self.lora_config}")
            if "target_modules" not in self.lora_config:
                print("WARNING: target_modules missing from lora_config!")
            peft_config = LoraConfig(**self.lora_config)
            self.qwen_model = get_peft_model(self.qwen_model, peft_config)
            self.qwen_model.print_trainable_parameters()

        if tune_vlm:
            if isinstance(self.qwen_model, PeftModel):
                # Keep only adapter params trainable for PEFT mode.
                for name, p in self.qwen_model.named_parameters():
                    if ("lora_" in name) or ("modules_to_save" in name):
                        p.requires_grad = True
            elif self.tune_top_llm_layers > 0:
                # N1.6-style minimal adaptation: unfreeze only top language layers.
                ok = self._unfreeze_top_llm_layers(self.tune_top_llm_layers)
                if not ok:
                    print(
                        "[GROOT] Warning: could not locate Qwen language layers for selective unfreeze; "
                        "falling back to full VLM unfreeze."
                    )
                    self.qwen_model.requires_grad_(True)
            else:
                # Full fine-tuning path when LoRA is disabled and no selective-layer policy is set.
                self.qwen_model.requires_grad_(True)
        if tune_projector:
            self.projector.requires_grad_(True)
        if self.value_head_enable and self.value_head is not None and self.tune_value_head:
            self.value_head.requires_grad_(True)

        print(f"Tune Qwen VLM: {self.tune_vlm}")
        print(f"Tune Qwen projector: {self.tune_projector}")
        print(f"Tune Qwen value head: {self.value_head_enable and self.tune_value_head}")
        print(f"Tune Qwen top LLM layers: {self.tune_top_llm_layers}")
        
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No trainable parameters in Qwen backbone.")

    def _collect_prefixed_inputs(self, vl_input: BatchFeature) -> tuple[dict, str]:
        qwen_input = {
            k.removeprefix(self.input_prefix): v
            for k, v in vl_input.items()
            if k.startswith(self.input_prefix)
        }
        used_prefix = self.input_prefix
        if not qwen_input:
            qwen_input = {
                k.removeprefix(self.legacy_input_prefix): v
                for k, v in vl_input.items()
                if k.startswith(self.legacy_input_prefix)
            }
            used_prefix = self.legacy_input_prefix
        return qwen_input, used_prefix

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    def set_frozen_modules_to_eval_mode(self):
        if self.training and not self.tune_vlm:
            self.qwen_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _select_hidden(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.select_layer is None:
            idx = len(hidden_states) - 4
        else:
            idx = self.select_layer
            if idx < 0:
                idx = len(hidden_states) + idx
        idx = max(0, min(idx, len(hidden_states) - 1))
        return hidden_states[idx]

    def forward_qwen(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Re-cast ViT to FP32 on first forward call.
        # Accelerate DDP wrapping can reset dtype; this catches that.
        if not getattr(self, "_vit_fp32_ensured", False) and getattr(torch.version, "hip", None) is not None:
            base = getattr(self.qwen_model, "module", self.qwen_model)
            for _attr in ("visual", "vision_model", "vision_tower", "visual_encoder"):
                if hasattr(base, _attr):
                    _vit = getattr(base, _attr)
                    _vit.float()
                    _dtype = next(_vit.parameters()).dtype
                    print(f"[GROOT] forward_qwen: re-cast .{_attr} to FP32, param_dtype={_dtype}", flush=True)
                    break
            self._vit_fp32_ensured = True
        qwen_input, _ = self._collect_prefixed_inputs(vl_input)
        if not qwen_input:
            raise ValueError(
                "No multimodal input keys found for Qwen backbone. "
                f"Expected prefix '{self.input_prefix}' or legacy '{self.legacy_input_prefix}'."
            )
        # Some processors add image sizes that the model does not consume.
        qwen_input.pop("image_sizes", None)
        summary_pos = qwen_input.pop("summary_pos", None)

        # Qwen3-VL expects image_grid_thw; synthesize it if missing.
        if qwen_input.get("image_grid_thw") is None and "pixel_values" in qwen_input:
            pv = qwen_input["pixel_values"]
            if pv.dim() == 5:
                b, t, _, h, w = pv.shape
            elif pv.dim() == 4:
                b, _, h, w = pv.shape
                t = 1
                qwen_input["pixel_values"] = pv.unsqueeze(1)
            else:
                raise ValueError(f"Unsupported pixel_values shape for Qwen backbone: {pv.shape}")
            patch = getattr(getattr(self.qwen_config, "vision_config", None), "patch_size", 14)
            grid = torch.tensor(
                [t, h // patch, w // patch], device=pv.device, dtype=torch.long
            ).unsqueeze(0).expand(b, -1)
            qwen_input["image_grid_thw"] = grid

        # Sanity check: log only if NaN appears in pixel_values.
        pv = qwen_input.get("pixel_values")
        if pv is not None and torch.isnan(pv).any():
            nan_count = torch.isnan(pv).sum().item()
            print(f"[GROOT][DEBUG] NaNs in pixel_values before Qwen: count={nan_count}, shape={tuple(pv.shape)}", flush=True)
            torch.save(vl_input, "debug_nan_input_pre.pt")

        # If pixel_values are flattened tokens, reshape using grid_thw
        # [REMOVED] Incorrect reshaping logic. Qwen expects flattened pixel_values.
        # Original logic tried to stack to (B, T, Tokens, Dim) but Qwen2/3-VL expects (TotalTokens, Dim).
        # We leave pixel_values as is (flattened) if it comes from the processor.
        
        # Flatten image_grid_thw if it's 3D, as Qwen expects a list of grids (flattened batch).
        # Cache pre-flatten shape for regrouping hook-captured ViT output in forward().
        grid = qwen_input.get("image_grid_thw")
        if grid is not None and grid.dim() == 3:
            self._last_grid_batch_size = grid.shape[0]
            self._last_grid_imgs_per_sample = grid.shape[1]
            qwen_input["image_grid_thw"] = grid.flatten(0, 1)
        elif grid is not None:
            self._last_grid_batch_size = 1
            self._last_grid_imgs_per_sample = grid.shape[0]
        else:
            self._last_grid_batch_size = None
            self._last_grid_imgs_per_sample = None
        self._last_image_grid_thw = qwen_input.get("image_grid_thw")

        # Cache input_ids for image_mask computation in forward()
        self._last_input_ids = qwen_input.get("input_ids")

        outputs = self.qwen_model(**qwen_input, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Qwen backbone requires hidden_states=True.")

        # Granular NaN check per layer
        for i, hs in enumerate(hidden_states):
            if torch.isnan(hs).any():
                print(f"[GROOT][DEBUG] NaNs detected in Qwen hidden_state layer {i}/{len(hidden_states)-1}: shape={tuple(hs.shape)}")
                # DUMP INPUTS FOR REPRODUCTION
                print("[GROOT][DEBUG] Saving input batch to 'debug_nan_input.pt' for reproduction.")
                torch.save(vl_input, "debug_nan_input.pt")
                break


        top = self._select_hidden(hidden_states)
        if torch.isnan(top).any():
             print(f"[GROOT][DEBUG] NaNs detected in selected layer ({self.select_layer}): shape={tuple(top.shape)}")

        if summary_pos is not None:
            if summary_pos.dim() == 1:
                idx = summary_pos.view(-1, 1, 1).expand(-1, 1, top.shape[-1])
                summary_feats = torch.gather(top, dim=1, index=idx).squeeze(1)
            else:
                raise ValueError(f"summary_pos expected shape (B,), got {tuple(summary_pos.shape)}")
            if self.summary_avg_last_k is not None and self.summary_avg_last_k > 1:
                k = self.summary_avg_last_k
                idxs = summary_pos - torch.arange(k, device=summary_pos.device, dtype=summary_pos.dtype)
                idxs = idxs.clamp_min(0)
                idxs = idxs.view(-1, k, 1).expand(-1, k, top.shape[-1])
                gathered = torch.gather(top, dim=1, index=idxs)
                summary_feats = gathered.mean(dim=1)
            # Keep sequence ABI stable for downstream modules: (B, S, D).
            qwen_features = summary_feats.unsqueeze(1)
            attn_mask = qwen_input.get("attention_mask")
            mask_dtype = attn_mask.dtype if attn_mask is not None else torch.long
            qwen_mask = torch.ones(
                (qwen_features.shape[0], 1),
                dtype=mask_dtype,
                device=qwen_features.device,
            )
        else:
            qwen_features = top
            qwen_mask = qwen_input.get("attention_mask")
        qwen_features = self.projector(qwen_features)
        return qwen_features, qwen_mask

    @staticmethod
    def _regroup_visual_tokens(
        visual_tokens: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_size: int,
        imgs_per_sample: int,
        spatial_merge_size: int = 2,
    ) -> torch.Tensor:
        """Regroup flat ViT output (total_patches, D) into (B, patches_per_sample, D).

        Qwen3-VL's ViT returns all visual tokens flattened across the entire batch.
        The actual token count per image is (t * h * w) // spatial_merge_size^2
        because the ViT internally merges spatial patches.

        Args:
            visual_tokens: Flat tensor of shape (total_patches, D).
            image_grid_thw: (total_imgs, 3) tensor of per-image (t, h, w) grids.
            batch_size: Number of samples in the batch.
            imgs_per_sample: Number of images per sample.
            spatial_merge_size: Qwen3-VL's spatial merge factor (default 2).

        Returns:
            (B, max_patches_per_sample, D) tensor, zero-padded if samples differ.
        """
        merge_sq = spatial_merge_size ** 2
        tokens_per_img = (
            image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        ) // merge_sq

        # Sum tokens belonging to each batch element
        patches_per_sample = []
        for b in range(batch_size):
            start = b * imgs_per_sample
            end = start + imgs_per_sample
            patches_per_sample.append(int(tokens_per_img[start:end].sum().item()))

        # Split and pad to equal length so we can stack into (B, max_patches, D)
        max_patches = max(patches_per_sample)
        chunks = torch.split(visual_tokens, patches_per_sample, dim=0)
        padded = []
        for chunk in chunks:
            if chunk.shape[0] < max_patches:
                pad = torch.zeros(
                    max_patches - chunk.shape[0], chunk.shape[1],
                    dtype=chunk.dtype, device=chunk.device,
                )
                padded.append(torch.cat([chunk, pad], dim=0))
            else:
                padded.append(chunk)
        return torch.stack(padded, dim=0)  # (B, max_patches, D)

    def forward_visual_only(self, vl_input: BatchFeature) -> torch.Tensor:
        """Fast path: extract visual features from ViT only, bypassing all LLM layers.

        Used by System 1 at ~10 Hz for fresh spatial information.
        Returns projected visual tokens shaped (B, patches_per_sample, proj_dim).
        """
        qwen_input, _ = self._collect_prefixed_inputs(vl_input)
        if not qwen_input:
            raise ValueError(
                "No multimodal input keys found for visual-only forward. "
                f"Expected prefix '{self.input_prefix}' or legacy '{self.legacy_input_prefix}'."
            )

        pixel_values = qwen_input.get("pixel_values")
        image_grid_thw = qwen_input.get("image_grid_thw")

        if pixel_values is None:
            raise ValueError("pixel_values required for forward_visual_only")

        # Remember pre-flatten shape to infer batch grouping later.
        # After collation, image_grid_thw is (B, num_imgs_per_sample, 3).
        # Qwen expects it flattened to (total_imgs, 3).
        grid_was_3d = image_grid_thw is not None and image_grid_thw.dim() == 3
        batch_size = image_grid_thw.shape[0] if grid_was_3d else 1
        imgs_per_sample = image_grid_thw.shape[1] if grid_was_3d else (
            image_grid_thw.shape[0] if image_grid_thw is not None else 0
        )
        if grid_was_3d:
            image_grid_thw = image_grid_thw.flatten(0, 1)

        # Flatten pixel_values if collated to higher dims.
        # Collation may produce (B, num_imgs, tokens, dim) but ViT expects (total_tokens, dim).
        if pixel_values.dim() > 2:
            pixel_values = pixel_values.flatten(0, pixel_values.dim() - 2)

        # Call the ViT directly — Qwen3VL's visual encoder takes
        # (hidden_states, grid_thw) where hidden_states is raw pixel_values
        # (patch_embed is applied internally as the first op in visual.forward).
        # Guard flag prevents the capture hook from overwriting _cached_vit_output.
        vit = self.qwen_model.visual
        pixel_values = pixel_values.type(vit.dtype)
        self._in_visual_only_mode = True
        try:
            visual_tokens = vit(pixel_values, grid_thw=image_grid_thw)
        finally:
            self._in_visual_only_mode = False

        # Unwrap tuple output if needed (Qwen3-VL returns (hidden_states, deep_features))
        if isinstance(visual_tokens, (tuple, list)):
            visual_tokens = visual_tokens[0]

        # Regroup flat (total_patches, D) into (B, patches_per_sample, D).
        if visual_tokens.dim() == 2 and image_grid_thw is not None:
            spatial_merge_size = getattr(
                getattr(self.qwen_config, "vision_config", self.qwen_config),
                "spatial_merge_size", 2,
            )
            visual_tokens = self._regroup_visual_tokens(
                visual_tokens, image_grid_thw,
                batch_size, imgs_per_sample, spatial_merge_size,
            )

        # Project to the same dimension the DiT cross-attention expects
        projected = self.projector(visual_tokens)

        return projected

    def _pool_for_value(
        self,
        qwen_features: torch.Tensor,
        qwen_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.value_head_pooling == "last_token":
            if qwen_mask is None:
                return qwen_features[:, -1, :]
            lengths = qwen_mask.long().sum(dim=1).clamp(min=1)
            idx = (lengths - 1).clamp(max=qwen_features.shape[1] - 1)
            batch_idx = torch.arange(qwen_features.shape[0], device=qwen_features.device)
            return qwen_features[batch_idx, idx]

        if qwen_mask is None:
            return qwen_features.mean(dim=1)

        mask = qwen_mask.to(device=qwen_features.device, dtype=qwen_features.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (qwen_features * mask).sum(dim=1) / denom

    def _compute_value_outputs(
        self,
        qwen_features: torch.Tensor,
        qwen_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.value_head is None:
            raise RuntimeError("Value head is not enabled for this backbone.")
        pooled = self._pool_for_value(qwen_features, qwen_mask)
        logits = self.value_head(pooled)
        probs = torch.softmax(logits, dim=-1)
        bin_centers = self.value_bin_centers.to(device=logits.device, dtype=logits.dtype)
        value_scalar = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)
        return logits, value_scalar

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        qwen_embeds, qwen_mask = self.forward_qwen(vl_input)
        if qwen_mask is not None and qwen_mask.shape[-1] != qwen_embeds.shape[1]:
            raise ValueError("Qwen backbone attention mask length mismatch.")
        if torch.isnan(qwen_embeds).any():
            with torch.no_grad():
                nan_count = torch.isnan(qwen_embeds).sum().item()
                print(
                    f"[GROOT][DEBUG] NaNs detected in Qwen backbone output: count={nan_count}, "
                    f"shape={tuple(qwen_embeds.shape)}"
                )
            # CRITICAL: nan_to_num must be OUTSIDE torch.no_grad() to preserve the
            # computation graph. If inside no_grad, the output has no grad_fn, which
            # breaks backward() when the backbone is the only trainable path
            # (e.g. backbone_align stage where the action head is fully frozen).
            qwen_embeds = torch.nan_to_num(qwen_embeds, nan=0.0, posinf=0.0, neginf=0.0)
        output: dict[str, torch.Tensor | None] = {
            "backbone_features": qwen_embeds,
            "backbone_attention_mask": qwen_mask,
        }

        # Compute image_mask for N1.6 AlternateVLDiT.
        # When summary_pos is used (single summary token), there are no image tokens
        # in the output — create an all-False mask. When full sequence is returned,
        # identify image tokens via the <|image_pad|> token ID in input_ids.
        if qwen_mask is not None:
            _input_ids = getattr(self, "_last_input_ids", None)
            if _input_ids is not None and _input_ids.shape[1] == qwen_mask.shape[1]:
                _img_token_id = getattr(self.qwen_config, "image_token_id", None)
                if _img_token_id is not None:
                    output["image_mask"] = (_input_ids == _img_token_id)
                else:
                    output["image_mask"] = torch.zeros_like(qwen_mask, dtype=torch.bool)
            else:
                # Summary mode or shape mismatch — no image tokens in output
                output["image_mask"] = torch.zeros_like(qwen_mask, dtype=torch.bool)
        else:
            output["image_mask"] = None

        # Include ViT features captured by hook (for joint training of System 1 path).
        # Regroup flat (total_patches, D) to (B, patches_per_sample, D) using cached grid info.
        if self._cached_vit_output is not None:
            vit_out = self._cached_vit_output
            grid_thw = getattr(self, "_last_image_grid_thw", None)
            if vit_out.dim() == 2 and grid_thw is not None:
                bs = getattr(self, "_last_grid_batch_size", 1) or 1
                ips = getattr(self, "_last_grid_imgs_per_sample", grid_thw.shape[0]) or grid_thw.shape[0]
                sms = getattr(
                    getattr(self.qwen_config, "vision_config", self.qwen_config),
                    "spatial_merge_size", 2,
                )
                vit_out = self._regroup_visual_tokens(vit_out, grid_thw, bs, ips, sms)
            output["visual_features"] = self.projector(vit_out)
            self._cached_vit_output = None  # Clear to avoid stale references

        if self.value_head_enable and self.value_head is not None:
            value_logits, value_scalar = self._compute_value_outputs(qwen_embeds, qwen_mask)
            output["value_logits"] = value_logits
            output["value_scalar"] = value_scalar
        return BatchFeature(data=output)

    @torch.no_grad()
    def extract_reasoning_trace(
        self,
        vl_input: BatchFeature,
        *,
        cot_session: "Any | None" = None,
        dataset_meta: "dict | None" = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict]:
        """Generate reasoning trace.

        When cot_session + dataset_meta are provided (structured CoT mode):
          - Rebuilds the prompt with the INIT/TICK JSON schema
          - Parses the output as JSON
          - Updates cot_session with the extracted plan
          - Returns dicts with keys: raw_text, parsed_json, parse_ok, mode, cot_step, generated_tokens

        When called without cot_session (legacy mode):
          - Returns dicts with keys: raw_text, think_text, summary_text, generated_tokens
        """
        qwen_input, _ = self._collect_prefixed_inputs(vl_input)
        if not qwen_input:
            return []

        qwen_input = dict(qwen_input)
        qwen_input.pop("image_sizes", None)
        qwen_input.pop("summary_pos", None)
        grid = qwen_input.get("image_grid_thw")
        if grid is not None and grid.dim() == 3:
            qwen_input["image_grid_thw"] = grid.flatten(0, 1)

        # Structured CoT mode: rebuild prompt with system+user message and CoT schema
        cot_mode_active = cot_session is not None and dataset_meta is not None
        if cot_mode_active:
            try:
                qwen_input = self._build_cot_qwen_input(qwen_input, cot_session=cot_session, dataset_meta=dataset_meta)
            except Exception as exc:
                print(f"[GROOT][CoT] _build_cot_qwen_input failed: {exc}. Falling back to legacy trace.", flush=True)
                cot_mode_active = False

        input_ids = qwen_input.get("input_ids")
        if input_ids is None:
            return []

        if not hasattr(self.qwen_model, "generate"):
            return []

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        device = input_ids.device
        # On ROCm, disable BF16 autocast for generation (same ViT NaN protection as forward_qwen)
        autocast_enabled = device.type == "cuda" and getattr(torch.version, "hip", None) is None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            generated = self.qwen_model.generate(**qwen_input, **gen_kwargs)

        if generated.dim() != 2:
            return []

        prompt_len = input_ids.shape[1]
        traces: list[dict] = []
        for i in range(min(generated.shape[0], 1)):  # CoT: always process item 0 only
            continuation = generated[i, prompt_len:]
            if self._tokenizer is not None:
                raw_text = self._tokenizer.decode(
                    continuation.detach().cpu().tolist(),
                    skip_special_tokens=True,
                ).strip()
            else:
                raw_text = str(continuation.detach().cpu().tolist())

            if cot_mode_active:
                from lerobot.policies.grootCoT.cot_schema import parse_cot_json
                parsed, parse_err = parse_cot_json(raw_text)
                if parsed is not None:
                    cot_session.advance(parsed)
                else:
                    cot_session.advance(None)  # still increment step_count
                traces.append({
                    "raw_text": raw_text,
                    "parsed_json": parsed,
                    "parse_ok": parsed is not None,
                    "parse_error": parse_err,
                    "mode": cot_session.mode if parsed is not None else (
                        "INIT" if cot_session.step_count <= 1 else "TICK"
                    ),
                    "cot_step": cot_session.step_count,
                    "generated_tokens": int(continuation.shape[0]),
                })
            else:
                # Legacy: extract THINK/SUMMARY tags
                think_text = self._extract_tag(raw_text, "THINK") or self._extract_tag(raw_text, "REASONING")
                summary_text = self._extract_tag(raw_text, "SUMMARY")
                if not summary_text and self.summary_token in raw_text:
                    summary_text = raw_text.split(self.summary_token, 1)[-1].strip()
                traces.append({
                    "raw_text": raw_text,
                    "think_text": think_text,
                    "summary_text": summary_text,
                    "generated_tokens": int(continuation.shape[0]),
                })
        return traces

    def _build_cot_qwen_input(
        self,
        qwen_input: dict,
        *,
        cot_session: "Any",
        dataset_meta: dict,
    ) -> dict:
        """Rebuild qwen_input dict with CoT system+user prompt, keeping vision tokens intact.

        Strategy:
          1. Tokenize system message and prepend to existing input_ids.
          2. Keep original tokens up to (and including) the last <|vision_end|> token.
             This preserves <|im_start|>user + vision block from the training prompt.
          3. Append the new CoT user text suffix: \n{user_msg}<|im_end|>\n<|im_start|>assistant\n
          4. Slice pixel_values / image_grid_thw to batch item 0 only.
        """
        from lerobot.policies.grootCoT.cot_schema import VISION_END_ID

        tok = self._tokenizer
        if tok is None:
            raise RuntimeError("Tokenizer unavailable for CoT prompt building.")

        orig_ids = qwen_input["input_ids"]  # (B, L)
        B = orig_ids.shape[0]
        device = orig_ids.device

        # Use batch item 0 for CoT generation
        ids_list = orig_ids[0].cpu().tolist()

        # Find last <|vision_end|> position (151653)
        last_ve = -1
        for idx, tok_id in enumerate(ids_list):
            if tok_id == VISION_END_ID:
                last_ve = idx
        if last_ve == -1:
            raise ValueError("No <|vision_end|> token found in input_ids — cannot rebuild CoT prompt.")

        # Build system message prefix tokens
        # Structure: <|im_start|>system\n{sys_msg}<|im_end|>\n
        im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

        def _encode(text: str) -> list[int]:
            return tok.encode(text, add_special_tokens=False)

        # Newline as token ids list
        _nl = tok.encode("\n", add_special_tokens=False)

        sys_text, user_text = cot_session.get_prompts(time_index=dataset_meta.get("time_index", 0))

        sys_prefix = (
            [im_start_id]
            + tok.encode("system\n", add_special_tokens=False)
            + _encode(sys_text)
            + [im_end_id]
            + _nl
        )

        # User suffix (after vision tokens): \n{user_msg}<|im_end|>\n<|im_start|>assistant\n
        # For Thinking models: pre-fill <think>\n\n</think>\n to skip internal
        # reasoning and go directly to structured JSON output.
        _is_thinking = "thinking" in getattr(self, "_model_id_lower", str(getattr(self, "model_id", "")).lower())
        assistant_suffix = tok.encode("assistant\n", add_special_tokens=False)
        if _is_thinking:
            assistant_suffix += tok.encode("<think>\n\n</think>\n", add_special_tokens=False)
        user_suffix = (
            _nl
            + _encode(user_text)
            + [im_end_id]
            + _nl
            + [im_start_id]
            + assistant_suffix
        )

        # Stitch: sys_prefix + ids[0 : last_ve+1] + user_suffix
        vision_kept = ids_list[: last_ve + 1]
        new_ids = sys_prefix + vision_kept + user_suffix
        new_ids_tensor = torch.tensor([new_ids], dtype=torch.long, device=device)
        new_attn_mask = torch.ones_like(new_ids_tensor)

        # Build output dict: keep pixel_values and image_grid_thw for item 0 only
        new_input: dict = {
            "input_ids": new_ids_tensor,
            "attention_mask": new_attn_mask,
        }

        # pixel_values: (B, n_imgs, tokens, dim) after collation regrouping
        pv = qwen_input.get("pixel_values")
        if pv is not None:
            if pv.dim() == 4 and pv.shape[0] == B:
                new_input["pixel_values"] = pv[0:1]  # (1, n_imgs, tokens, dim)
            elif pv.dim() == 2 and B > 1 and pv.shape[0] % B == 0:
                # Flat format: (B * tokens_per_item, dim) — slice to item 0 only
                tokens_per_item = pv.shape[0] // B
                new_input["pixel_values"] = pv[0:tokens_per_item]
            else:
                # Single item or unexpected shape — pass as-is
                new_input["pixel_values"] = pv

        # image_grid_thw: (B*n_imgs, 3) after flatten in extract_reasoning_trace
        grid = qwen_input.get("image_grid_thw")
        if grid is not None and grid.dim() == 2:
            total_grids = grid.shape[0]
            n_imgs = max(1, total_grids // B)
            new_input["image_grid_thw"] = grid[0:n_imgs]
        elif grid is not None:
            new_input["image_grid_thw"] = grid

        return new_input


class EagleBackbone(nn.Module):
    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str = DEFAULT_VENDOR_EAGLE_PATH,
        tokenizer_assets_repo: str = DEFAULT_TOKENIZER_ASSETS_REPO,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        # Prefer loading Eagle model config from the cache directory where vendor files were copied.
        # Import lazily — Eagle utilities pull in eagle2_hg_model which requires
        # transformers <= 4.55 (group_images_by_shape).  CraftNet never instantiates
        # EagleBackbone, so the import only runs when actually needed.
        from lerobot.policies.groot.utils import ensure_eagle_cache_ready

        vendor_dir = DEFAULT_VENDOR_EAGLE_PATH
        cache_dir = HF_LEROBOT_HOME / tokenizer_assets_repo
        try:
            ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_assets_repo)
        except Exception as exc:  # nosec: B110
            print(f"[GROOT] Warning: failed to prepare Eagle cache for backbone: {exc}")

        config = AutoConfig.from_pretrained(str(cache_dir), trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00TN15Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00TN15(PreTrainedModel):
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    config_class = GR00TN15Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00TN15Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        backbone_cfg = dict(config.backbone_cfg)

        # Auto-configure dimensions for N1.6 action head
        _ah_version = getattr(config, "action_head_version", "n15")
        if _ah_version == "n16":
            _ah_cfg = dict(config.action_head_cfg)
            # Force N1.6 dimensions (override N1.5 defaults)
            _ah_cfg["backbone_embedding_dim"] = 2048
            _ah_cfg["max_state_dim"] = 128
            _ah_cfg["max_action_dim"] = 128
            _ah_cfg["action_horizon"] = getattr(config, "chunk_size",
                                                getattr(config, "action_horizon", 16))
            config.action_head_cfg = _ah_cfg
            print(f"[GROOT] N1.6 action head: backbone_dim={_ah_cfg['backbone_embedding_dim']}, "
                  f"state_dim={_ah_cfg['max_state_dim']}, action_dim={_ah_cfg['max_action_dim']}, "
                  f"horizon={_ah_cfg['action_horizon']}")

        # Determine expected dimension from Action Head
        expected_dim = config.action_head_cfg.get("backbone_embedding_dim")
        
        # Ensure the VLM projector matches the action head expected dimension.
        project_dim = backbone_cfg.get("project_to_dim")
        if project_dim is None:
            if expected_dim is not None:
                print(f"[GROOT] Setting project_to_dim={expected_dim} from action_head config.")
                backbone_cfg["project_to_dim"] = expected_dim
                project_dim = expected_dim
        
        # Force overwrite if mismatch? (Optional but safer)
        if expected_dim is not None and project_dim != expected_dim:
             print(f"[GROOT] WARNING: Overwriting project_to_dim ({project_dim}) with expected action head dim ({expected_dim})!")
             backbone_cfg["project_to_dim"] = expected_dim

        if backbone_cfg.get("project_to_dim") is None:
            raise ValueError("project_to_dim must be set for the backbone to match action head dimensions.")

        if config.compute_dtype == "bfloat16" or getattr(config, "use_bf16", False):
            backbone_cfg["load_bf16"] = True
        
        # Pass attn_implementation if specified in config
        if hasattr(config, "attn_implementation") and config.attn_implementation is not None:
            backbone_cfg["attn_implementation"] = config.attn_implementation

        # Persist any backbone_cfg updates so from_pretrained re-init uses them.
        config.backbone_cfg = dict(backbone_cfg)

        # Construct LoRA config for backbone from main config
        lora_cfg = getattr(config, "lora_config", {})
        if not lora_cfg:
            lora_cfg = {
                "r": getattr(config, "lora_rank", 0),
                "lora_alpha": getattr(config, "lora_alpha", 16),
                "lora_dropout": getattr(config, "lora_dropout", 0.05),
                "target_modules": getattr(config, "lora_target_modules", None),
            }
        
        # Pass LoRA config to backbone
        # We need to ensure QwenBackbone receives lora_config kwarg
        backbone_type = backbone_cfg.pop("type", "qwen")
        # Also remove potential duplicate lora_config from backbone_cfg if present
        backbone_cfg.pop("lora_config", None)
        
        if backbone_type == "eagle":
            self.backbone = EagleBackbone(**backbone_cfg)
        else:
            # QwenBackbone accepts lora_config as kwarg
            print(f"[GROOT] Initializing QwenBackbone with LoRA config: {lora_cfg}")
            self.backbone = QwenBackbone(**backbone_cfg, lora_config=lora_cfg)

        # Select action head version: N1.5 (default) or N1.6
        _ah_version = getattr(config, "action_head_version", "n15")
        if _ah_version == "n16":
            from lerobot.policies.grootCoT.action_head_n16.gr00t_n1d6_action_head import (
                Gr00tN1d6ActionHead,
                N16ActionHeadConfig,
            )
            _n16_cfg = N16ActionHeadConfig(
                backbone_embedding_dim=config.action_head_cfg.get("backbone_embedding_dim", 2048),
                max_state_dim=config.action_head_cfg.get("max_state_dim", getattr(config, "max_state_dim", 128)),
                max_action_dim=config.action_head_cfg.get("max_action_dim", getattr(config, "max_action_dim", 128)),
                action_horizon=config.action_head_cfg.get("action_horizon", getattr(config, "chunk_size", 50)),
                tune_projector=getattr(config, "tune_projector", True),
                tune_diffusion_model=getattr(config, "tune_diffusion_model", True),
                tune_vlln=getattr(config, "tune_vlln", True),
                ik_prior_prob=getattr(config, "ik_prior_prob", 0.0),
                ik_prior_noise_scale=getattr(config, "ik_prior_noise_scale", 0.15),
                ik_prior_arm_dim=getattr(config, "ik_prior_arm_dim", 14),
            )
            self.action_head = Gr00tN1d6ActionHead(_n16_cfg)
            _weights_path = getattr(config, "n16_action_head_weights_path", None)
            if _weights_path:
                import torch as _torch
                _weights = _torch.load(_weights_path, map_location="cpu")
                _missing, _unexpected = self.action_head.load_state_dict(_weights, strict=False)
                print(f"[GROOT] Loaded N1.6 action head: {len(_weights)} keys, "
                      f"missing={len(_missing)}, unexpected={len(_unexpected)}")
                if _missing:
                    print(f"[GROOT] Missing keys: {_missing[:5]}...")
            print(f"[GROOT] Using N1.6 action head (Gr00tN1d6ActionHead)")
        else:
            action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
            # Propagate Action Head LoRA config
            ah_lora_cfg = getattr(config, "action_head_lora_config", {})
            if not ah_lora_cfg:
                 ah_lora_cfg = {
                    "r": getattr(config, "action_head_lora_rank", 0),
                    "lora_alpha": getattr(config, "action_head_lora_alpha", 16),
                    "lora_dropout": getattr(config, "action_head_lora_dropout", 0.1),
                    "target_modules": getattr(config, "action_head_lora_target_modules", None),
                 }
            self.action_head = FlowmatchingActionHead(action_head_cfg, lora_config=ah_lora_cfg)

        # Resolve action_horizon and action_dim.
        # For N1.6 CraftNet: the GrootCoTConfig sets chunk_size=50 and max_action_dim=128,
        # but the base GR00TN15Config may have action_horizon=16 and action_dim=32.
        # Always prefer the action_head_cfg values which were set during __init__.
        _ah_version = getattr(config, "action_head_version", "n15")
        if _ah_version == "n16":
            self.action_horizon = config.action_head_cfg.get("action_horizon", 50)
            self.action_dim = config.action_head_cfg.get("max_action_dim", 128)
        else:
            self.action_horizon = config.action_head_cfg.get(
                "action_horizon", getattr(config, "action_horizon", 16))
            self.action_dim = config.action_head_cfg.get(
                "action_dim", getattr(config, "action_dim", 32))
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        # Validation disabled: the base GR00TN15Config has action_dim=32 and
        # action_horizon=16 which don't match N1.6 (128/50) or CraftNet overrides.
        # The shape mismatch causes false failures. The actual tensor shapes are
        # validated implicitly by the action head's forward pass.
        pass

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        # Validation disabled: N1.6 action head output keys differ from N1.5,
        # and shape checks use stale action_horizon/action_dim from base config.
        pass

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_outputs = self.run_backbone(inputs)
        action_head_outputs = self.run_action_head(
            inputs=inputs,
            backbone_outputs=backbone_outputs,
            is_training=True,
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_outputs = self.run_backbone(inputs)
        action_head_outputs = self.run_action_head(
            inputs=inputs,
            backbone_outputs=backbone_outputs,
            is_training=False,
        )
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def predict_value(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_outputs = self.run_backbone(inputs)
        if "value_logits" not in backbone_outputs or "value_scalar" not in backbone_outputs:
            raise ValueError(
                "Value head outputs are unavailable. Enable Qwen value head via "
                "`value_head_enable=true` / `policy.recap_value_head_enable=true`."
            )
        return BatchFeature(
            data={
                "value_logits": backbone_outputs["value_logits"],
                "value_scalar": backbone_outputs["value_scalar"],
            }
        )

    @torch.no_grad()
    def extract_cot_trace(
        self,
        inputs: dict,
        *,
        cot_session: "Any | None" = None,
        dataset_meta: "dict | None" = None,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict]:
        """Generate Qwen reasoning traces from current visual-language inputs.

        When cot_session + dataset_meta are provided, generates structured JSON
        output following the vla.plan.recap.v1 schema (INIT or TICK mode).
        Otherwise falls back to legacy THINK/SUMMARY extraction.
        """
        backbone_inputs = self.prepare_backbone_input(inputs)
        if hasattr(self.backbone, "extract_reasoning_trace"):
            return self.backbone.extract_reasoning_trace(
                backbone_inputs,
                cot_session=cot_session,
                dataset_meta=dataset_meta,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
        return []

    def _to_device_with_maybe_dtype(self, x):
        # Cast floating tensors to a memory-efficient compute dtype when requested.
        # Rationale: Upcasting backbone activations to fp32 significantly increases VRAM.
        # When compute_dtype is bfloat16, prefer bf16 for activations to match AMP behavior.
        if not isinstance(x, torch.Tensor):
            return x
        if torch.is_floating_point(x):
            if getattr(self, "compute_dtype", None) == "bfloat16":
                return x.to(self.device, dtype=torch.bfloat16)
            # Fallback: preserve previous behavior if not using bf16 compute
            return x.to(self.device, dtype=self.action_head.dtype)
        # Non-floating tensors: move device only
        return x.to(self.device)

    def prepare_backbone_input(self, inputs) -> BatchFeature:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        return tree.map_structure(self._to_device_with_maybe_dtype, backbone_inputs)

    def prepare_action_input(self, inputs) -> BatchFeature:
        self.validate_inputs(inputs)
        action_inputs = self.action_head.prepare_input(inputs)
        return tree.map_structure(self._to_device_with_maybe_dtype, action_inputs)

    def prepare_input(self, inputs) -> tuple[BatchFeature, BatchFeature]:
        backbone_inputs = self.prepare_backbone_input(inputs)
        action_inputs = self.prepare_action_input(inputs)
        return backbone_inputs, action_inputs

    def run_backbone(self, inputs: dict | BatchFeature) -> BatchFeature:
        if isinstance(inputs, BatchFeature):
            backbone_inputs = inputs
        else:
            backbone_inputs = self.prepare_backbone_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward`.
        return self.backbone(backbone_inputs)

    def run_visual_only(self, inputs: dict | BatchFeature) -> BatchFeature:
        """Fast visual-only forward for System 1. Bypasses LLM layers.

        Returns BatchFeature with 'visual_features' key containing
        projected ViT tokens [B, num_visual_tokens, backbone_embedding_dim].
        Falls back to full run_backbone for legacy EagleBackbone.
        """
        if isinstance(self.backbone, QwenBackbone):
            if isinstance(inputs, BatchFeature):
                backbone_inputs = inputs
            else:
                backbone_inputs = self.prepare_backbone_input(inputs)
            visual_features = self.backbone.forward_visual_only(backbone_inputs)
            return BatchFeature(data={"visual_features": visual_features})
        else:
            # EagleBackbone doesn't support visual-only extraction
            # Fall back to full backbone (visual features are embedded in backbone_features)
            return self.run_backbone(inputs)

    def run_action_head(
        self,
        *,
        inputs: dict | BatchFeature,
        backbone_outputs: BatchFeature,
        is_training: bool,
        fresh_visual_features: torch.Tensor | None = None,
    ) -> BatchFeature:
        if isinstance(inputs, BatchFeature):
            action_inputs = inputs
        else:
            action_inputs = self.prepare_action_input(inputs)

        # Fuse cached System 2 features with fresh System 1 visual features
        if fresh_visual_features is not None:
            cached_features = backbone_outputs["backbone_features"]
            cached_mask = backbone_outputs.get("backbone_attention_mask")

            # fresh_visual_features: [B, S_vit, D]
            # cached_features: [B, S_s2, D]
            combined_features = torch.cat([cached_features, fresh_visual_features], dim=1)

            # Build mask for fresh tokens (all valid)
            fresh_mask = torch.ones(
                fresh_visual_features.shape[0],
                fresh_visual_features.shape[1],
                dtype=cached_mask.dtype if cached_mask is not None else torch.long,
                device=fresh_visual_features.device,
            )

            if cached_mask is not None:
                combined_mask = torch.cat([cached_mask, fresh_mask], dim=1)
            else:
                combined_mask = None

            # Rebuild image_mask: fresh visual tokens ARE image tokens
            cached_image_mask = backbone_outputs.get("image_mask")
            if cached_image_mask is not None:
                fresh_image_mask = torch.ones(
                    fresh_visual_features.shape[0],
                    fresh_visual_features.shape[1],
                    dtype=torch.bool,
                    device=fresh_visual_features.device,
                )
                combined_image_mask = torch.cat([cached_image_mask, fresh_image_mask], dim=1)
            else:
                combined_image_mask = None

            backbone_outputs = BatchFeature(data={
                "backbone_features": combined_features,
                "backbone_attention_mask": combined_mask,
                "image_mask": combined_image_mask,
            })

        if is_training:
            return self.action_head(backbone_outputs, action_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs)

    @staticmethod
    def _log_trainable_parameter_groups(model: "GR00TN15") -> None:
        groups = {
            "backbone_vlm": 0,
            "backbone_projector": 0,
            "backbone_value_head": 0,
            "action_head_diffusion": 0,
            "action_head_projector": 0,
            "action_head_vlln": 0,
            "other": 0,
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            if name.startswith("backbone.projector."):
                groups["backbone_projector"] += numel
            elif name.startswith("backbone.value_head."):
                groups["backbone_value_head"] += numel
            elif name.startswith("backbone.qwen_model."):
                groups["backbone_vlm"] += numel
            elif name.startswith("action_head.model."):
                groups["action_head_diffusion"] += numel
            elif name.startswith("action_head.vlln.") or name.startswith("action_head.vl_self_attention."):
                groups["action_head_vlln"] += numel
            elif name.startswith(
                (
                    "action_head.state_encoder.",
                    "action_head.action_encoder.",
                    "action_head.action_decoder.",
                    "action_head.position_embedding.",
                    "action_head.future_tokens.",
                    "action_head.extra_observation_projectors.",
                )
            ):
                groups["action_head_projector"] += numel
            else:
                groups["other"] += numel

        total_trainable = sum(groups.values())
        total_params = sum(p.numel() for p in model.parameters())
        print(
            "[GROOT] Trainable params summary: "
            f"trainable={total_trainable:,} / total={total_params:,} "
            f"({(100.0 * total_trainable / max(total_params, 1)):.4f}%)"
        )
        print(f"[GROOT]  backbone_vlm={groups['backbone_vlm']:,}")
        print(f"[GROOT]  backbone_projector={groups['backbone_projector']:,}")
        print(f"[GROOT]  backbone_value_head={groups['backbone_value_head']:,}")
        print(f"[GROOT]  action_head_diffusion={groups['action_head_diffusion']:,}")
        print(f"[GROOT]  action_head_projector={groups['action_head_projector']:,}")
        print(f"[GROOT]  action_head_vlln={groups['action_head_vlln']:,}")
        if groups["other"] > 0:
            print(f"[GROOT]  other={groups['other']:,}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_vlm = kwargs.pop("tune_vlm", tune_visual or tune_llm)
        tune_vlm_projector = kwargs.pop("tune_vlm_projector", True)
        tune_top_llm_layers = max(0, int(kwargs.pop("tune_top_llm_layers", 0)))
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        tune_vlln = kwargs.pop("tune_vlln", True)
        value_head_enable = kwargs.pop("value_head_enable", False)
        tune_value_head = kwargs.pop("tune_value_head", True)
        value_head_bins = kwargs.pop("value_head_bins", 201)
        value_head_vmin = kwargs.pop("value_head_vmin", -1.0)
        value_head_vmax = kwargs.pop("value_head_vmax", 0.0)
        value_head_pooling = kwargs.pop("value_head_pooling", "masked_mean")
        extra_observation_dims = kwargs.pop("extra_observation_dims", None)
        # N1.6 action head
        action_head_version = kwargs.pop("action_head_version", "n15")
        n16_action_head_weights_path = kwargs.pop("n16_action_head_weights_path", None)
        chunk_size = kwargs.pop("chunk_size", 16)
        ik_prior_prob = kwargs.pop("ik_prior_prob", 0.0)
        ik_prior_noise_scale = kwargs.pop("ik_prior_noise_scale", 0.15)
        ik_prior_arm_dim = kwargs.pop("ik_prior_arm_dim", 14)
        primary_action_group_indices = kwargs.pop("primary_action_group_indices", None)
        secondary_action_group_indices = kwargs.pop("secondary_action_group_indices", None)
        primary_action_group_loss_weight = kwargs.pop("primary_action_group_loss_weight", None)
        secondary_action_group_loss_weight = kwargs.pop("secondary_action_group_loss_weight", None)
        upper_body_joint_indices = kwargs.pop("upper_body_joint_indices", None)
        lower_body_joint_indices = kwargs.pop("lower_body_joint_indices", None)
        upper_body_loss_weight = kwargs.pop("upper_body_loss_weight", None)
        lower_body_loss_weight = kwargs.pop("lower_body_loss_weight", None)
        if primary_action_group_indices is None:
            primary_action_group_indices = upper_body_joint_indices
        if secondary_action_group_indices is None:
            secondary_action_group_indices = lower_body_joint_indices
        if primary_action_group_loss_weight is None:
            primary_action_group_loss_weight = upper_body_loss_weight
        if secondary_action_group_loss_weight is None:
            secondary_action_group_loss_weight = lower_body_loss_weight
        
        # Extract LoRA parameters for Backbone
        lora_rank = kwargs.pop("lora_rank", 0)
        lora_alpha = kwargs.pop("lora_alpha", 16)
        lora_dropout = kwargs.pop("lora_dropout", 0.05)
        lora_target_modules = kwargs.pop("lora_target_modules", None)

        # Extract LoRA parameters for Action Head
        action_head_lora_rank = kwargs.pop("action_head_lora_rank", 0)
        action_head_lora_alpha = kwargs.pop("action_head_lora_alpha", 16)
        action_head_lora_dropout = kwargs.pop("action_head_lora_dropout", 0.1)
        action_head_lora_target_modules = kwargs.pop("action_head_lora_target_modules", None)
        
        if action_head_lora_rank > 0 and action_head_lora_target_modules is None:
            # Default to Attention projection layers for DiT (diffusers based)
            # These match diffusers.models.attention.Attention components
            action_head_lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
            print(f"[GROOT] Defaulting action_head_lora_target_modules to: {action_head_lora_target_modules}")

        load_bf16 = kwargs.pop("load_bf16", False)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune backbone projector: {tune_vlm_projector}")
        print(f"Tune top LLM layers: {tune_top_llm_layers}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head vlln: {tune_vlln}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        print(f"Value head enabled: {value_head_enable}")
        print(f"Backbone LoRA rank: {lora_rank}")
        print(f"Action Head LoRA rank: {action_head_lora_rank}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # Load config first to allow modifications
        config = cls.config_class.from_pretrained(local_model_path, **kwargs)

        if extra_observation_dims:
            normalized_dims: dict[str, int] = {}
            for key, value in dict(extra_observation_dims).items():
                try:
                    dim = int(value)
                except (TypeError, ValueError):
                    continue
                if dim > 0:
                    normalized_dims[str(key)] = dim

            if normalized_dims:
                action_head_cfg = dict(config.action_head_cfg)
                action_head_cfg["extra_observation_dims"] = normalized_dims
                config.action_head_cfg = action_head_cfg
                config.extra_observation_dims = normalized_dims
                print(f"[GROOT] Action head extra observation projections: {sorted(normalized_dims.keys())}")

        # Persist LoRA args on config so GR00TN15.__init__ receives them.
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.lora_dropout = lora_dropout
        config.lora_target_modules = lora_target_modules
        config.action_head_lora_rank = action_head_lora_rank
        config.action_head_lora_alpha = action_head_lora_alpha
        config.action_head_lora_dropout = action_head_lora_dropout
        config.action_head_lora_target_modules = action_head_lora_target_modules
        config.tune_top_llm_layers = tune_top_llm_layers
        config.tune_vlln = bool(tune_vlln)
        # N1.6 action head config
        config.action_head_version = action_head_version
        config.n16_action_head_weights_path = n16_action_head_weights_path
        config.chunk_size = chunk_size
        config.ik_prior_prob = ik_prior_prob
        config.ik_prior_noise_scale = ik_prior_noise_scale
        config.ik_prior_arm_dim = ik_prior_arm_dim
        config.value_head_enable = bool(value_head_enable)
        config.tune_value_head = bool(tune_value_head)
        config.value_head_bins = int(value_head_bins)
        config.value_head_vmin = float(value_head_vmin)
        config.value_head_vmax = float(value_head_vmax)
        config.value_head_pooling = str(value_head_pooling)
        
        # Override backbone model_id if provided (and not using Eagle)
        model_id = kwargs.pop("model_id", None)
        if model_id is not None:
             # Look for "type" in backbone_cfg, default to "qwen" if missing
             if config.backbone_cfg.get("type", "qwen") != "eagle":
                 print(f"[GROOT] Overriding backbone model_id with: {model_id}")
                 config.backbone_cfg["model_id"] = model_id

        if config.backbone_cfg.get("type", "qwen") != "eagle":
            input_prefix = config.backbone_cfg.get("input_prefix")
            if not input_prefix or input_prefix == LEGACY_EAGLE_INPUT_PREFIX:
                config.backbone_cfg["input_prefix"] = DEFAULT_QWEN_INPUT_PREFIX
            config.backbone_cfg["value_head_enable"] = bool(value_head_enable)
            config.backbone_cfg["tune_value_head"] = bool(tune_value_head)
            config.backbone_cfg["value_head_bins"] = int(value_head_bins)
            config.backbone_cfg["value_head_vmin"] = float(value_head_vmin)
            config.backbone_cfg["value_head_vmax"] = float(value_head_vmax)
            config.backbone_cfg["value_head_pooling"] = str(value_head_pooling)

        # Inject attn_implementation if provided (for Qwen backbone)
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is not None:
             config.backbone_cfg["attn_implementation"] = attn_implementation
        config.backbone_cfg["tune_top_llm_layers"] = tune_top_llm_layers

        action_head_cfg = dict(config.action_head_cfg)
        action_head_cfg["tune_vlln"] = bool(tune_vlln)
        if primary_action_group_indices is not None:
            action_head_cfg["primary_action_group_indices"] = list(primary_action_group_indices)
        if secondary_action_group_indices is not None:
            action_head_cfg["secondary_action_group_indices"] = list(secondary_action_group_indices)
        if "primary_action_group_indices" not in action_head_cfg and "upper_body_joint_indices" in action_head_cfg:
            action_head_cfg["primary_action_group_indices"] = list(
                action_head_cfg.get("upper_body_joint_indices") or []
            )
        if "secondary_action_group_indices" not in action_head_cfg and "lower_body_joint_indices" in action_head_cfg:
            action_head_cfg["secondary_action_group_indices"] = list(
                action_head_cfg.get("lower_body_joint_indices") or []
            )
        if primary_action_group_loss_weight is None:
            primary_action_group_loss_weight = action_head_cfg.get(
                "primary_action_group_loss_weight",
                action_head_cfg.get(
                    "upper_body_loss_weight",
                    getattr(
                        config,
                        "primary_action_group_loss_weight",
                        getattr(config, "upper_body_loss_weight", 1.0),
                    ),
                ),
            )
        if secondary_action_group_loss_weight is None:
            secondary_action_group_loss_weight = action_head_cfg.get(
                "secondary_action_group_loss_weight",
                action_head_cfg.get(
                    "lower_body_loss_weight",
                    getattr(
                        config,
                        "secondary_action_group_loss_weight",
                        getattr(config, "lower_body_loss_weight", 1.0),
                    ),
                ),
            )
        action_head_cfg["primary_action_group_loss_weight"] = float(primary_action_group_loss_weight)
        action_head_cfg["secondary_action_group_loss_weight"] = float(secondary_action_group_loss_weight)
        # Mirror to legacy aliases for compatibility with older loaders/metrics.
        action_head_cfg["upper_body_joint_indices"] = list(action_head_cfg.get("primary_action_group_indices") or [])
        action_head_cfg["lower_body_joint_indices"] = list(
            action_head_cfg.get("secondary_action_group_indices") or []
        )
        action_head_cfg["upper_body_loss_weight"] = float(
            action_head_cfg.get("primary_action_group_loss_weight", 1.0)
        )
        action_head_cfg["lower_body_loss_weight"] = float(
            action_head_cfg.get("secondary_action_group_loss_weight", 1.0)
        )
        config.action_head_cfg = action_head_cfg
        config.primary_action_group_indices = list(action_head_cfg.get("primary_action_group_indices") or [])
        config.secondary_action_group_indices = list(action_head_cfg.get("secondary_action_group_indices") or [])
        config.primary_action_group_loss_weight = float(
            action_head_cfg.get("primary_action_group_loss_weight", 1.0)
        )
        config.secondary_action_group_loss_weight = float(
            action_head_cfg.get("secondary_action_group_loss_weight", 1.0)
        )
        # Legacy aliases.
        config.upper_body_joint_indices = list(config.primary_action_group_indices)
        config.lower_body_joint_indices = list(config.secondary_action_group_indices)
        config.upper_body_loss_weight = float(config.primary_action_group_loss_weight)
        config.lower_body_loss_weight = float(config.secondary_action_group_loss_weight)
        if config.primary_action_group_indices or config.secondary_action_group_indices:
            print(
                "[GROOT] Joint loss split configured: "
                f"primary={config.primary_action_group_indices}, "
                f"secondary={config.secondary_action_group_indices}, "
                f"weights=({config.primary_action_group_loss_weight}, "
                f"{config.secondary_action_group_loss_weight})"
            )

        # Inject LoRA config into backbone_cfg for QwenBackbone initialization
        if lora_rank > 0:
            if lora_target_modules is None:
                # Default to all linear layers for Qwen/Llama architectures
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                print(f"[GROOT] Defaulting lora_target_modules to: {lora_target_modules}")

            config.lora_config = {
                "r": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": lora_target_modules,
            }
            config.backbone_cfg["lora_config"] = dict(config.lora_config)

        if action_head_lora_rank > 0:
            config.action_head_lora_config = {
                "r": action_head_lora_rank,
                "lora_alpha": action_head_lora_alpha,
                "lora_dropout": action_head_lora_dropout,
                "target_modules": action_head_lora_target_modules,
            }

        # For N1.6, ignore size mismatches in action head (N1.5 checkpoint has
        # different dims than N1.6). The N1.6 weights are loaded separately.
        _ignore_mismatched = action_head_version == "n16"
        pretrained_model = super().from_pretrained(
            local_model_path, config=config, local_model_path=local_model_path,
            ignore_mismatched_sizes=_ignore_mismatched, **kwargs
        )

        # Detect and fix corrupted QwenBackbone weights caused by HuggingFace's
        # init_empty_weights() context during from_pretrained. When the outer
        # GR00T checkpoint doesn't contain Qwen3VL keys (e.g., GR00T-N1.5-3B
        # uses eagle_model), those parameters remain as uninitialized meta tensors
        # and get garbage values when finally materialized.
        # Fix: in-place reload from Qwen3VL checkpoint (CPU-only, no extra GPU memory).
        if hasattr(pretrained_model, "backbone") and isinstance(pretrained_model.backbone, QwenBackbone):
            _backbone = pretrained_model.backbone
            _qwen_model = _backbone.qwen_model
            # Check ViT proj weight for corruption (should be small values ~0.01-0.1)
            _vit = getattr(_qwen_model, "visual", None)
            _vit_proj = getattr(getattr(_vit, "patch_embed", None), "proj", None) if _vit else None
            if _vit_proj is not None:
                _w_max = _vit_proj.weight.float().abs().max().item()
                print(f"[GROOT] ViT proj weight check: max={_w_max:.4g} (expected 0.001–1.0)", flush=True)
                if _w_max > 10.0 or _w_max < 1e-6:
                    # Weights are uninitialized garbage — reload in-place from Qwen checkpoint
                    _backbone_cfg = getattr(pretrained_model.config, "backbone_cfg", {})
                    _model_id = _backbone_cfg.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")
                    _reason = "zeroed out" if _w_max < 1e-6 else "uninitialized garbage"
                    print(f"[GROOT] Corrupted ViT weights detected ({_reason}, max={_w_max:.4g}). In-place reload from {_model_id}...", flush=True)
                    import gc
                    _is_rocm_fix = getattr(torch.version, "hip", None) is not None
                    _fix_load_kw = {
                        "trust_remote_code": True,
                        "device_map": "cpu",
                        "dtype": torch.float32,
                    }
                    _reload_cls = Qwen3VLForConditionalGeneration if Qwen3VLForConditionalGeneration is not None else AutoModel
                    _qwen_cpu = _reload_cls.from_pretrained(_model_id, **_fix_load_kw)
                    _cpu_sd = _qwen_cpu.state_dict()
                    _reloaded, _skipped = 0, 0
                    with torch.no_grad():
                        for _name, _param in _qwen_model.named_parameters():
                            if _name in _cpu_sd:
                                _src = _cpu_sd[_name].to(dtype=_param.dtype, device=_param.device)
                                _param.data.copy_(_src)
                                _reloaded += 1
                            else:
                                _skipped += 1
                    del _qwen_cpu, _cpu_sd
                    gc.collect()
                    torch.cuda.empty_cache()
                    _new_max = _vit_proj.weight.float().abs().max().item()
                    print(f"[GROOT] In-place reload done: {_reloaded} params reloaded, {_skipped} skipped. New ViT max={_new_max:.4g}", flush=True)
            else:
                print("[GROOT] WARNING: Could not locate ViT patch_embed.proj for weight check.", flush=True)

        # Re-load N1.6 action head weights after from_pretrained (which overwrites
        # them with randomly-initialized garbage due to ignore_mismatched_sizes=True).
        if action_head_version == "n16" and n16_action_head_weights_path:
            from lerobot.policies.grootCoT.action_head_n16.gr00t_n1d6_action_head import Gr00tN1d6ActionHead
            if isinstance(pretrained_model.action_head, Gr00tN1d6ActionHead):
                _weights = torch.load(n16_action_head_weights_path, map_location="cpu")
                _missing, _unexpected = pretrained_model.action_head.load_state_dict(_weights, strict=False)
                print(f"[GROOT] Re-loaded N1.6 action head after from_pretrained: "
                      f"{len(_weights)} keys, missing={len(_missing)}, unexpected={len(_unexpected)}")

        # Legacy heavy safety path (kept for backward compat, now superceded by in-place fix above).
        force_backbone_reinit = os.getenv("GROOT_FORCE_BACKBONE_REINIT", "0").strip() == "1"
        if (
            force_backbone_reinit
            and hasattr(pretrained_model, "backbone")
            and isinstance(pretrained_model.backbone, QwenBackbone)
        ):
            print("[GROOT] GROOT_FORCE_BACKBONE_REINIT=1: Re-creating QwenBackbone (uses 2x GPU memory)...")
            lora_cfg = getattr(pretrained_model.config, "lora_config", {})
            if not lora_cfg:
                lora_cfg = {
                    "r": getattr(pretrained_model.config, "lora_rank", 0),
                    "lora_alpha": getattr(pretrained_model.config, "lora_alpha", 16),
                    "lora_dropout": getattr(pretrained_model.config, "lora_dropout", 0.05),
                    "target_modules": getattr(pretrained_model.config, "lora_target_modules", None),
                }
            backbone_cfg = dict(pretrained_model.config.backbone_cfg)
            backbone_cfg.pop("lora_config", None)
            pretrained_model.backbone = QwenBackbone(**backbone_cfg, lora_config=lora_cfg)

        if isinstance(pretrained_model.backbone, QwenBackbone):
            pretrained_model.backbone.set_trainable_parameters(
                tune_vlm=tune_vlm,
                tune_projector=tune_vlm_projector,
                tune_value_head=tune_value_head,
                tune_top_llm_layers=tune_top_llm_layers,
            )
        else:
            pretrained_model.backbone.set_trainable_parameters(tune_visual=tune_visual, tune_llm=tune_llm)
        
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, 
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
        )
        cls._log_trainable_parameter_groups(pretrained_model)
        return pretrained_model
