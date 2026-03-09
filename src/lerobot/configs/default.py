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

from dataclasses import dataclass, field

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


UNITREE_DEX3_DATASET_REPOS: dict[str, str] = {
    "blockstacking": "unitreerobotics/G1_Dex3_BlockStacking_Dataset",
    "objectplacement": "unitreerobotics/G1_Dex3_ObjectPlacement_Dataset",
    "graspsquare": "unitreerobotics/G1_Dex3_GraspSquare_Dataset",
    "pickapple": "unitreerobotics/G1_Dex3_PickApple_Dataset",
    "pickbottle": "unitreerobotics/G1_Dex3_PickBottle_Dataset",
    "pickcharger": "unitreerobotics/G1_Dex3_PickCharger_Dataset",
    "pickdoll": "unitreerobotics/G1_Dex3_PickDoll_Dataset",
    "pickgum": "unitreerobotics/G1_Dex3_PickGum_Dataset",
    "picksnack": "unitreerobotics/G1_Dex3_PickSnack_Dataset",
    "picktissue": "unitreerobotics/G1_Dex3_PickTissue_Dataset",
    "pouring": "unitreerobotics/G1_Dex3_Pouring_Dataset",
    "camerapackaging": "unitreerobotics/G1_Dex3_CameraPackaging_Dataset",
    "toastedbread": "unitreerobotics/G1_Dex3_ToastedBread_Dataset",
}

UNITREE_DEX3_ALL_ALIASES = {
    "all",
    "alldex3",
    "allunitree",
    "unitreeall",
    "dex3all",
}


def _normalize_dataset_alias(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str | list[str] = ""
    # Optional selector for official Unitree Dex3 datasets hosted on Hugging Face.
    # Supports:
    # - short aliases like "block_stacking", "pick_bottle", "camera_packaging"
    # - full repo IDs, e.g. "unitreerobotics/G1_Dex3_BlockStacking_Dataset"
    # - "all" to train on all official Unitree Dex3 datasets in a single run
    # - comma-separated aliases/repo IDs, e.g. "block_stacking,pouring,pick_bottle"
    # If provided, it overrides `repo_id`.
    dex3_dataset: str | None = None
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    # Optional local Humanoid-Everyday zip inputs.
    # If set, dataset loading will use these zip files directly instead of Hugging Face parquet files.
    # Example: `--dataset.zip_paths='["~/Downloads/task_a.zip","~/Downloads/task_b.zip"]'`
    zip_paths: list[str] | None = None
    # Optional glob pattern for Humanoid-Everyday zip files.
    # Example: `--dataset.zip_glob="~/Downloads/humanoid_everyday_g1_basic_zips/*.zip"`
    zip_glob: str | None = None
    episodes: list[int] | None = None
    # Optional exact task-name filter. Matches against full names like "Basic/stack_two_cubes_g1"
    # and bare names like "stack_two_cubes_g1".
    task_names: list[str] | None = None
    # Optional task-category filter. Supports both dataset categories (e.g. "Basic")
    # and task-summary aliases (e.g. "basic_manipulation", "articulate_object").
    task_categories: list[str] | None = None
    # Optional robot type filter (e.g. ["g1"] or ["h1"]).
    robot_types: list[str] | None = None
    # Optional keyword-based episode filtering using episode-level task strings.
    # Example: `["dex3", "brainco"]` keeps episodes whose task text contains either keyword.
    task_include_keywords: list[str] | None = None
    # Optional negative filter applied after include filter.
    task_exclude_keywords: list[str] | None = None
    task_keyword_case_sensitive: bool = False
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False
    # Allow overriding the tolerance used when aligning timestamps to frame
    # indices. Defaults to the canonical 1e-4 seconds.
    tolerance_s: float = 1e-4

    def _resolve_single_dex3_selector(self, selector: str) -> str:
        selected = str(selector).strip()
        if "/" in selected:
            return selected

        alias = _normalize_dataset_alias(selected)
        if alias in UNITREE_DEX3_DATASET_REPOS:
            return UNITREE_DEX3_DATASET_REPOS[alias]

        valid_aliases = sorted(UNITREE_DEX3_DATASET_REPOS)
        raise ValueError(
            "Unknown dataset.dex3_dataset value: "
            f"'{selector}'. Expected one of aliases={valid_aliases}, "
            "a comma-separated list of them, 'all', or a full HF dataset repo_id."
        )

    def resolve_repo_id(self) -> str | list[str]:
        if self.dex3_dataset is None or str(self.dex3_dataset).strip() == "":
            if isinstance(self.repo_id, list):
                return [str(repo).strip() for repo in self.repo_id if str(repo).strip()]
            return str(self.repo_id or "").strip()

        selected = str(self.dex3_dataset).strip()
        normalized = _normalize_dataset_alias(selected)
        if normalized in UNITREE_DEX3_ALL_ALIASES:
            return list(UNITREE_DEX3_DATASET_REPOS.values())

        normalized_csv = selected.replace(";", ",").replace("+", ",")
        selectors = [part.strip() for part in normalized_csv.split(",") if part.strip()]
        if len(selectors) > 1:
            resolved: list[str] = []
            seen: set[str] = set()
            for selector in selectors:
                repo = self._resolve_single_dex3_selector(selector)
                if repo not in seen:
                    resolved.append(repo)
                    seen.add(repo)
            return resolved

        return self._resolve_single_dex3_selector(selected)


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class PeftConfig:
    """Optional PEFT override config used by train pipeline when --peft.* CLI args are provided.

    PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    effective methods so we'll focus on those in this high-level config interface.
    """

    peft_type: str | None = None
    task_type: str | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    bias: str | None = None

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None
    modules_to_save: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
