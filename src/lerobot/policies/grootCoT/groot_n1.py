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
        Qwen3VLForConditionalGeneration,
        PretrainedConfig,
        PreTrainedModel,
    )
    from transformers.feature_extraction_utils import BatchFeature
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
from lerobot.policies.groot.utils import ensure_eagle_cache_ready
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
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_SUMMARY_TOKEN = "<SUMMARY>"
DEFAULT_QWEN_INPUT_PREFIX = "qwen_"
LEGACY_EAGLE_INPUT_PREFIX = "eagle_"


class QwenBackbone(nn.Module):
    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_MODEL_ID,
        tune_vlm: bool = False,
        tune_projector: bool = True,
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
        self._tokenizer = None

        dtype = torch.bfloat16 if load_bf16 else None
        self.qwen_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        load_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        # Prefer the multimodal VL class; fall back to generic vision2seq, then causal LM, then base AutoModel.
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
                     # Last resort: Qwen3VL if strictly needed for Qwen2.5-VL compatibility in older transformers?
                     # But current issue is using Qwen3 for Qwen2.
                     if Qwen3VLForConditionalGeneration is not None:
                         self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
                     else:
                         self.qwen_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        
        print(f"[GROOT] Initialized QwenBackbone with model class: {type(self.qwen_model)}")
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
            # Middle layer (skip embeddings at index 0) to retain visual info
            self.select_layer = min(max(1, 1 + num_layers // 2), num_layers)

        # Truncate layers to save compute and memory (matching original Groot implementation)
        if self.select_layer is not None and num_layers is not None:
             # Qwen2/3-VL structure: qwen_model.model.layers
             if hasattr(self.qwen_model, "model") and hasattr(self.qwen_model.model, "layers"):
                 layers = self.qwen_model.model.layers
                 while len(layers) > self.select_layer:
                     # Remove the last layer until we match select_layer
                     layers.pop(-1)
                 print(f"Truncated Qwen backbone to {len(layers)} layers (select_layer={self.select_layer})")

        if project_to_dim is None:
            self.projector = nn.Identity()
        else:
            self.projector = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, project_to_dim))
            if load_bf16:
                self.projector = self.projector.to(torch.bfloat16)

        self.set_trainable_parameters(
            tune_vlm=tune_vlm,
            tune_projector=tune_projector,
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
        tune_top_llm_layers: int | None = None,
    ):
        self.tune_vlm = tune_vlm
        self.tune_projector = tune_projector
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

        print(f"Tune Qwen VLM: {self.tune_vlm}")
        print(f"Tune Qwen projector: {self.tune_projector}")
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
            idx = len(hidden_states) // 2
        else:
            idx = self.select_layer
            if idx < 0:
                idx = len(hidden_states) + idx
        idx = max(0, min(idx, len(hidden_states) - 1))
        return hidden_states[idx]

    def forward_qwen(self, vl_input: BatchFeature) -> tuple[torch.Tensor, torch.Tensor | None]:
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

        # Debug input stats to trace NaNs/shape issues.
        pv = qwen_input.get("pixel_values")
        if pv is not None:
            if torch.isnan(pv).any():
                nan_count = torch.isnan(pv).sum().item()
                print(f"[GROOT][DEBUG] NaNs in pixel_values before Qwen: count={nan_count}, shape={tuple(pv.shape)}")
                torch.save(vl_input, "debug_nan_input_pre.pt")
                print("Saved debug_nan_input_pre.pt")
            else:
                with torch.no_grad():
                    print(
                        "[GROOT][DEBUG] pixel_values stats: "
                        f"shape={tuple(pv.shape)}, min={pv.min().item()}, max={pv.max().item()}, mean={pv.mean().item()}, dtype={pv.dtype}"
                    )
        grid = qwen_input.get("image_grid_thw")
        if grid is not None:
            if torch.isnan(grid.float()).any():
                nan_count = torch.isnan(grid.float()).sum().item()
                print(f"[GROOT][DEBUG] NaNs in image_grid_thw before Qwen: count={nan_count}, shape={tuple(grid.shape)}")
            else:
                print(f"[GROOT][DEBUG] image_grid_thw: shape={tuple(grid.shape)}, values={grid[0].tolist() if grid.numel() else 'empty'}")

        # If pixel_values are flattened tokens, reshape using grid_thw
        # [REMOVED] Incorrect reshaping logic. Qwen expects flattened pixel_values.
        # Original logic tried to stack to (B, T, Tokens, Dim) but Qwen2/3-VL expects (TotalTokens, Dim).
        # We leave pixel_values as is (flattened) if it comes from the processor.
        
        # Flatten image_grid_thw if it's 3D, as Qwen expects a list of grids (flattened batch)
        grid = qwen_input.get("image_grid_thw")
        if grid is not None and grid.dim() == 3:
            qwen_input["image_grid_thw"] = grid.flatten(0, 1)

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
                qwen_embeds = torch.nan_to_num(qwen_embeds, nan=0.0, posinf=0.0, neginf=0.0)
        return BatchFeature(data={"backbone_features": qwen_embeds, "backbone_attention_mask": qwen_mask})

    @torch.no_grad()
    def extract_reasoning_trace(
        self,
        vl_input: BatchFeature,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict[str, str | int]]:
        """Generate CoT-like text and extract THINK/SUMMARY fields."""
        qwen_input, _ = self._collect_prefixed_inputs(vl_input)
        if not qwen_input:
            return []

        qwen_input = dict(qwen_input)
        qwen_input.pop("image_sizes", None)
        qwen_input.pop("summary_pos", None)
        grid = qwen_input.get("image_grid_thw")
        if grid is not None and grid.dim() == 3:
            qwen_input["image_grid_thw"] = grid.flatten(0, 1)

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
        autocast_enabled = device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            generated = self.qwen_model.generate(**qwen_input, **gen_kwargs)

        if generated.dim() != 2:
            return []

        prompt_len = input_ids.shape[1]
        traces: list[dict[str, str | int]] = []
        for i in range(generated.shape[0]):
            continuation = generated[i, prompt_len:]
            if self._tokenizer is not None:
                raw_text = self._tokenizer.decode(
                    continuation.detach().cpu().tolist(),
                    skip_special_tokens=True,
                ).strip()
            else:
                raw_text = str(continuation.detach().cpu().tolist())

            think_text = self._extract_tag(raw_text, "THINK") or self._extract_tag(raw_text, "REASONING")
            summary_text = self._extract_tag(raw_text, "SUMMARY")
            if not summary_text and self.summary_token in raw_text:
                summary_text = raw_text.split(self.summary_token, 1)[-1].strip()

            traces.append(
                {
                    "raw_text": raw_text,
                    "think_text": think_text,
                    "summary_text": summary_text,
                    "generated_tokens": int(continuation.shape[0]),
                }
            )
        return traces


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

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            # In inference, action may be omitted or None; validate only when it's a tensor.
            if action is None:
                pass  # allow None during inference
            elif isinstance(action, torch.Tensor):
                shape_ok = (
                    len(action.shape) == 3
                    and action.shape[1] == self.action_horizon
                    and action.shape[2] == self.action_dim
                )
                if not shape_ok:
                    error_msg += f"\n{action.shape=}"
                    detected_error = True
            else:
                # Unexpected non-tensor type provided for action
                error_msg += f"\nInvalid type for action: {type(action)}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature) or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

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

    @torch.no_grad()
    def extract_cot_trace(
        self,
        inputs: dict,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[dict[str, str | int]]:
        """Generate Qwen reasoning traces from current visual-language inputs."""
        backbone_inputs = self.prepare_backbone_input(inputs)
        if hasattr(self.backbone, "extract_reasoning_trace"):
            return self.backbone.extract_reasoning_trace(
                backbone_inputs,
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

    def run_action_head(
        self,
        *,
        inputs: dict | BatchFeature,
        backbone_outputs: BatchFeature,
        is_training: bool,
    ) -> BatchFeature:
        if isinstance(inputs, BatchFeature):
            action_inputs = inputs
        else:
            action_inputs = self.prepare_action_input(inputs)
        if is_training:
            return self.action_head(backbone_outputs, action_inputs)
        return self.action_head.get_action(backbone_outputs, action_inputs)

    @staticmethod
    def _log_trainable_parameter_groups(model: "GR00TN15") -> None:
        groups = {
            "backbone_vlm": 0,
            "backbone_projector": 0,
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
        extra_observation_dims = kwargs.pop("extra_observation_dims", None)
        upper_body_joint_indices = kwargs.pop("upper_body_joint_indices", None)
        lower_body_joint_indices = kwargs.pop("lower_body_joint_indices", None)
        upper_body_loss_weight = kwargs.pop("upper_body_loss_weight", None)
        lower_body_loss_weight = kwargs.pop("lower_body_loss_weight", None)
        
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

        # Inject attn_implementation if provided (for Qwen backbone)
        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is not None:
             config.backbone_cfg["attn_implementation"] = attn_implementation
        config.backbone_cfg["tune_top_llm_layers"] = tune_top_llm_layers

        action_head_cfg = dict(config.action_head_cfg)
        action_head_cfg["tune_vlln"] = bool(tune_vlln)
        if upper_body_joint_indices is not None:
            action_head_cfg["upper_body_joint_indices"] = list(upper_body_joint_indices)
        if lower_body_joint_indices is not None:
            action_head_cfg["lower_body_joint_indices"] = list(lower_body_joint_indices)
        if upper_body_loss_weight is None:
            upper_body_loss_weight = action_head_cfg.get(
                "upper_body_loss_weight",
                getattr(config, "upper_body_loss_weight", 1.0),
            )
        if lower_body_loss_weight is None:
            lower_body_loss_weight = action_head_cfg.get(
                "lower_body_loss_weight",
                getattr(config, "lower_body_loss_weight", 1.0),
            )
        action_head_cfg["upper_body_loss_weight"] = float(upper_body_loss_weight)
        action_head_cfg["lower_body_loss_weight"] = float(lower_body_loss_weight)
        config.action_head_cfg = action_head_cfg
        config.upper_body_joint_indices = list(action_head_cfg.get("upper_body_joint_indices") or [])
        config.lower_body_joint_indices = list(action_head_cfg.get("lower_body_joint_indices") or [])
        config.upper_body_loss_weight = float(action_head_cfg.get("upper_body_loss_weight", 1.0))
        config.lower_body_loss_weight = float(action_head_cfg.get("lower_body_loss_weight", 1.0))
        if config.upper_body_joint_indices or config.lower_body_joint_indices:
            print(
                "[GROOT] Joint loss split configured: "
                f"upper={config.upper_body_joint_indices}, "
                f"lower={config.lower_body_joint_indices}, "
                f"weights=({config.upper_body_loss_weight}, {config.lower_body_loss_weight})"
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

        pretrained_model = super().from_pretrained(
            local_model_path, config=config, local_model_path=local_model_path, **kwargs
        )

        # Re-initialize Qwen Backbone to ensure no contamination from GR00T checkpoint
        # (e.g. key collisions with Eagle backbone in the checkpoint)
        if hasattr(pretrained_model, "backbone") and isinstance(pretrained_model.backbone, QwenBackbone):
             print("[GROOT] Re-initializing Qwen Backbone to purge potential checkpoint contamination...")
             # Re-create backbone using the configuration
             # Note: config.backbone_cfg is a dict
             # Rebuild LoRA config for Qwen backbone (if any)
             lora_cfg = getattr(pretrained_model.config, "lora_config", {})
             if not lora_cfg:
                 lora_cfg = {
                     "r": getattr(pretrained_model.config, "lora_rank", 0),
                     "lora_alpha": getattr(pretrained_model.config, "lora_alpha", 16),
                     "lora_dropout": getattr(pretrained_model.config, "lora_dropout", 0.05),
                     "target_modules": getattr(pretrained_model.config, "lora_target_modules", None),
                 }
             backbone_cfg = dict(pretrained_model.config.backbone_cfg)
             # Avoid passing lora_config twice (via backbone_cfg and explicit kwarg).
             backbone_cfg.pop("lora_config", None)
             pretrained_model.backbone = QwenBackbone(**backbone_cfg, lora_config=lora_cfg)
             # Move to device and dtype matching the model
             # (Actually the model is likely on CPU or meta device here if loaded via from_pretrained?)
             # But we need it to match.
             # AutoModel loading in QwenBackbone handles device placement usually? 
             # No, it loads to default. We should let accelerate handle device placement later.
             # One nuance: if load_bf16 was used, QwenBackbone init handles it.

        if isinstance(pretrained_model.backbone, QwenBackbone):
            pretrained_model.backbone.set_trainable_parameters(
                tune_vlm=tune_vlm,
                tune_projector=tune_vlm_projector,
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
