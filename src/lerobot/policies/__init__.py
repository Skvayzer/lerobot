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

from __future__ import annotations

import importlib


def _optional_import(module: str, attr: str) -> object | None:
    try:
        mod = importlib.import_module(module, package=__name__)
        return getattr(mod, attr)
    except Exception:
        return None


ACTConfig = _optional_import(".act.configuration_act", "ACTConfig")
DiffusionConfig = _optional_import(".diffusion.configuration_diffusion", "DiffusionConfig")
GrootConfig = _optional_import(".groot.configuration_groot", "GrootConfig")
GrootCoTConfig = _optional_import(".grootCoT.configuration_groot", "GrootCoTConfig")
PI0Config = _optional_import(".pi0.configuration_pi0", "PI0Config")
PI0FastConfig = _optional_import(".pi0_fast.configuration_pi0_fast", "PI0FastConfig")
PI05Config = _optional_import(".pi05.configuration_pi05", "PI05Config")
SmolVLAConfig = _optional_import(".smolvla.configuration_smolvla", "SmolVLAConfig")
SmolVLANewLineProcessor = _optional_import(".smolvla.processor_smolvla", "SmolVLANewLineProcessor")
SARMConfig = _optional_import(".sarm.configuration_sarm", "SARMConfig")
TDMPCConfig = _optional_import(".tdmpc.configuration_tdmpc", "TDMPCConfig")
VQBeTConfig = _optional_import(".vqbet.configuration_vqbet", "VQBeTConfig")
WallXConfig = _optional_import(".wall_x.configuration_wall_x", "WallXConfig")
XVLAConfig = _optional_import(".xvla.configuration_xvla", "XVLAConfig")

__all__ = [
    name
    for name in [
        "ACTConfig",
        "DiffusionConfig",
        "PI0Config",
        "PI05Config",
        "PI0FastConfig",
        "SmolVLAConfig",
        "SmolVLANewLineProcessor",
        "SARMConfig",
        "TDMPCConfig",
        "VQBeTConfig",
        "GrootConfig",
        "GrootCoTConfig",
        "XVLAConfig",
        "WallXConfig",
    ]
    if globals().get(name) is not None
]

