#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH --job-name=groot_n16_dex1sim_v2
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n16_dex1sim_v2_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n16_dex1sim_v2_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "GR00T N1.6 Dex1 Sim BlockStacking v2 (4 GPU, stats fix)"
echo "Model: nvidia/GR00T-N1.6-3B (32-layer AlternateVLDiT)"
echo "Fixes: reinject_dataset_stats at creation + save, correct eval processors"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

source /vast/users/chenyuan.chen/miniconda3/bin/activate unitree_lerobot_amd

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/vast/users/chenyuan.chen/.cache/huggingface
export HF_LEROBOT_HOME=/vast/users/chenyuan.chen/.cache/huggingface/lerobot
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Install gr00t package into training env if not already installed.
echo "Checking gr00t package installation..."
if ! python -c "import gr00t" 2>/dev/null; then
    echo "Installing gr00t package (--no-deps) from /vast/users/chenyuan.chen/Isaac-GR00T ..."
    pip install --no-deps -e /vast/users/chenyuan.chen/Isaac-GR00T
    pip install \
        diffusers==0.35.1 \
        peft==0.17.1 \
        einops==0.8.1 \
        gymnasium==1.2.2 \
        omegaconf==2.3.0 \
        lmdb==1.7.5 \
        msgpack==1.1.0 \
        "msgpack-numpy==0.4.8" \
        "albumentations==1.4.18" \
        "dm-tree==0.1.8" \
        termcolor==3.2.0 \
        "tyro==0.9.17" \
        "gitpython==3.1.46" \
        "pyzmq==27.0.1" \
        2>&1 | grep -v "^Requirement already"
    echo "gr00t installed."
else
    echo "gr00t already installed: $(python -c 'import gr00t; print(gr00t.__version__)')"
fi

# Pin transformers to 4.55.0 (compatible with Eagle2.5/Eagle3 and HybridCache)
CURRENT_TF=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "Current transformers: $CURRENT_TF"
if [ "$CURRENT_TF" != "4.55.0" ]; then
    echo "Installing transformers==4.55.0 (current: $CURRENT_TF) ..."
    pip install 'transformers==4.55.0' 2>&1 | tail -5
    echo "transformers now: $(python -c 'import transformers; print(transformers.__version__)')"
else
    echo "transformers 4.55.0, OK"
fi

# Patch Eagle3_VL model for ROCm
GROOT_MODULES_DIR=/vast/users/chenyuan.chen/Isaac-GR00T/gr00t/model/modules
HF_EAGLE_CACHE=/vast/users/chenyuan.chen/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2
python3 - <<'PYEOF'
import os, re

TARGETS = [
    os.environ.get(
        "GROOT_MODULES_DIR",
        "/vast/users/chenyuan.chen/Isaac-GR00T/gr00t/model/modules"
    ) + "/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py",
    os.environ.get(
        "HF_EAGLE_CACHE",
        "/vast/users/chenyuan.chen/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2"
    ) + "/modeling_eagle3_vl.py",
]

ASSERT_PATTERN = re.compile(
    r'assert \(\s*config\.text_config\._attn_implementation == "flash_attention_2"\s*\)'
    r',\s*f"(Qwen[23]) must use flash_attention_2 but got \{config\.text_config\._attn_implementation\}"',
    re.DOTALL,
)
ASSERT_REPLACEMENT = (
    r'pass  # ROCm compat: relaxed flash_attention_2 assertion for \1'
)

for path in TARGETS:
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        continue
    with open(path) as f:
        content = f.read()
    patched = ASSERT_PATTERN.sub(ASSERT_REPLACEMENT, content)
    patched = patched.replace(
        'config.vision_config._attn_implementation = "flash_attention_2"',
        'config.vision_config._attn_implementation = "eager"',
    )
    if patched == content:
        print(f"Already patched or no match: {path}")
    else:
        with open(path, "w") as f:
            f.write(patched)
        print(f"Patched Eagle3_VL assertions: {path}")

VIDEO_INPUT_OLD = (
    "from transformers.image_utils import (\n"
    "    ImageInput,\n"
    "    VideoInput,\n"
    "    get_image_size,\n"
    "    to_numpy_array,\n"
    ")"
)
VIDEO_INPUT_NEW = (
    "from transformers.image_utils import (\n"
    "    ImageInput,\n"
    "    get_image_size,\n"
    "    to_numpy_array,\n"
    ")\n"
    "try:\n"
    "    from transformers.image_utils import VideoInput\n"
    "except ImportError:\n"
    "    VideoInput = ImageInput  # transformers compat fallback"
)
PROC_PATHS = [
    os.environ.get("GROOT_MODULES_DIR", "/vast/users/chenyuan.chen/Isaac-GR00T/gr00t/model/modules")
    + "/nvidia/Eagle-Block2A-2B-v2/processing_eagle3_vl.py",
    os.environ.get("HF_EAGLE_CACHE", "/vast/users/chenyuan.chen/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2")
    + "/processing_eagle3_vl.py",
]
for PROC_PATH in PROC_PATHS:
    if not os.path.exists(PROC_PATH):
        print(f"SKIP (not found): {PROC_PATH}")
        continue
    with open(PROC_PATH) as f:
        content = f.read()
    if VIDEO_INPUT_OLD in content:
        with open(PROC_PATH, "w") as f:
            f.write(content.replace(VIDEO_INPUT_OLD, VIDEO_INPUT_NEW))
        print(f"Patched VideoInput import: {PROC_PATH}")
    elif "VideoInput = ImageInput" in content:
        print(f"VideoInput already patched: {PROC_PATH}")
    else:
        print(f"WARN: VideoInput pattern not found in {PROC_PATH}")

IMG_FAST_OLD_FAST = (
    "from transformers.image_processing_utils_fast import (\n"
    "    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,\n"
    "    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,\n"
    "    BaseImageProcessorFast,\n"
    "    DefaultFastImageProcessorKwargs,\n"
    "    divide_to_patches,\n"
    "    group_images_by_shape,\n"
    "    reorder_images,\n"
    ")"
)
IMG_FAST_NEW_FAST = (
    "from transformers.image_processing_utils_fast import (\n"
    "    BaseImageProcessorFast,\n"
    "    DefaultFastImageProcessorKwargs,\n"
    "    divide_to_patches,\n"
    "    group_images_by_shape,\n"
    "    reorder_images,\n"
    ")\n"
    "try:\n"
    "    from transformers.image_processing_utils_fast import (\n"
    "        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,\n"
    "        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,\n"
    "    )\n"
    "except ImportError:\n"
    "    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ''  # compat stub\n"
    "    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ''  # compat stub"
)
IMG_FAST_OLD_UTILS = (
    "from transformers.image_utils import (\n"
    "    OPENAI_CLIP_MEAN,\n"
    "    OPENAI_CLIP_STD,\n"
    "    IMAGENET_STANDARD_MEAN,  # 0.5, 0.5, 0.5\n"
    "    IMAGENET_STANDARD_STD,  # 0.5, 0.5, 0.5\n"
    "    ChannelDimension,\n"
    "    ImageInput,\n"
    "    VideoInput,\n"
    "    PILImageResampling,\n"
    "    SizeDict,\n"
    "    get_image_size,\n"
    "    make_flat_list_of_images,\n"
    "    make_batched_videos,\n"
    "    validate_kwargs,\n"
    ")"
)
IMG_FAST_NEW_UTILS = (
    "from transformers.image_utils import (\n"
    "    OPENAI_CLIP_MEAN,\n"
    "    OPENAI_CLIP_STD,\n"
    "    IMAGENET_STANDARD_MEAN,  # 0.5, 0.5, 0.5\n"
    "    IMAGENET_STANDARD_STD,  # 0.5, 0.5, 0.5\n"
    "    ChannelDimension,\n"
    "    ImageInput,\n"
    "    PILImageResampling,\n"
    "    SizeDict,\n"
    "    get_image_size,\n"
    "    make_flat_list_of_images,\n"
    "    validate_kwargs,\n"
    ")\n"
    "try:\n"
    "    from transformers.image_utils import VideoInput\n"
    "except ImportError:\n"
    "    VideoInput = ImageInput  # compat\n"
    "try:\n"
    "    from transformers.image_utils import make_batched_videos\n"
    "except ImportError:\n"
    "    def make_batched_videos(videos):  # compat stub\n"
    "        return [videos] if not isinstance(videos, list) else videos"
)
IMG_FAST_PATHS = [
    os.environ.get("GROOT_MODULES_DIR", "/vast/users/chenyuan.chen/Isaac-GR00T/gr00t/model/modules")
    + "/nvidia/Eagle-Block2A-2B-v2/image_processing_eagle3_vl_fast.py",
    os.environ.get("HF_EAGLE_CACHE", "/vast/users/chenyuan.chen/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2")
    + "/image_processing_eagle3_vl_fast.py",
]
for fpath in IMG_FAST_PATHS:
    if not os.path.exists(fpath):
        print(f"SKIP (not found): {fpath}")
        continue
    with open(fpath) as f:
        content = f.read()
    changed = False
    if IMG_FAST_OLD_FAST in content:
        content = content.replace(IMG_FAST_OLD_FAST, IMG_FAST_NEW_FAST)
        changed = True
    if IMG_FAST_OLD_UTILS in content:
        content = content.replace(IMG_FAST_OLD_UTILS, IMG_FAST_NEW_UTILS)
        changed = True
    if changed:
        with open(fpath, "w") as f:
            f.write(content)
        print(f"Patched image_processing_eagle3_vl_fast.py: {fpath}")
    else:
        print(f"Already patched or no match: {os.path.basename(fpath)}")
PYEOF
export GROOT_MODULES_DIR HF_EAGLE_CACHE

CACHE_ROOT=/tmp/$USER/rocm_cache_${SLURM_JOB_ID}
mkdir -p "$CACHE_ROOT"/{miopen_db,miopen_cache,torch_kernels,xdg_cache}
chmod -R u+rwX "$CACHE_ROOT"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen_cache"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1

NUM_GPUS=4
echo "GPUs: $NUM_GPUS | Conda: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

git pull origin unitree-features || echo "WARNING: git pull failed, continuing"

# Pre-download the dataset
echo "Pre-downloading Dex1 Sim dataset..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_LEROBOT_HOME']
snapshot_download('unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim', repo_type='dataset', local_dir=f'{cache}/unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim')
print('Download complete')
" || echo "WARNING: pre-download failed, training will attempt download on-the-fly"

# Pre-download GR00T N1.6 model weights
echo "Pre-downloading GR00T N1.6 model..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.6-3B', repo_type='model')
print('GR00T N1.6 download complete')
" || echo "WARNING: GR00T N1.6 pre-download failed, will download on-the-fly"

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_groot_n16_dex1sim_blockstack_v2

accelerate launch \
  --multi_gpu --num_processes="$NUM_GPUS" --num_machines=1 --mixed_precision=no \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="$OUTPUT_DIR" \
  --policy.type=groot_n16 \
  --policy.base_model_path=nvidia/GR00T-N1.6-3B \
  --policy.embodiment_tag=unitree_g1 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  --policy.max_state_dim=128 \
  --policy.max_action_dim=128 \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.use_bf16=true \
  --policy.use_flash_attention=false \
  --policy.push_to_hub=false \
  --dataset.repo_id=unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim \
  --dataset.root="$HF_LEROBOT_HOME" \
  --dataset.robot_types='["g1"]' \
  --dataset.video_backend=pyav \
  --tolerance_s=5e-4 \
  --dataset.tolerance_s=5e-4 \
  --batch_size=8 \
  --num_workers=2 \
  --epochs=5 \
  --save_freq=2000 \
  --keep_last_n_checkpoints=2 \
  --wandb.enable=true \
  --wandb.project=G1_Groot_Baselines \
  --wandb.entity=skvayzer \
  --wandb.disable_artifact=true \
  --wandb.notes="GR00T-N1.6 Dex1 Sim v2: 4GPU, reinject_dataset_stats fix, correct eval processors"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
