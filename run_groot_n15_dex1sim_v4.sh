#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH --job-name=groot_n15_dex1sim_v5
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n15_dex1sim_v5_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n15_dex1sim_v5_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "GR00T N1.5 Dex1 Sim BlockStacking v5 (4 GPU, stats fix)"
echo "Fixes: reinject_dataset_stats, eval processors, Eagle2.5 fast processor compat"
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

# Patch HF-cached Eagle2.5 processor for transformers 4.51.3 compat
# Fixes: (1) VideoInput import, (2) _prepare_image_like_inputs (added in 4.53)
echo "Patching Eagle2.5 HF cache for transformers 4.51.3 compat..."
python3 - <<'EAGLE_PATCH'
import os, glob

# All directories that may contain Eagle2.5 processor files
SEARCH_DIRS = []
for base in [
    os.path.expanduser("~/.cache/huggingface/modules/transformers_modules"),
    os.path.expanduser("~/.cache/huggingface/lerobot/lerobot"),
]:
    for pattern in ["eagle2hg-processor-groot-n1p5", "eagle2hg*"]:
        SEARCH_DIRS.extend(glob.glob(os.path.join(base, pattern)))

for d in SEARCH_DIRS:
    if not os.path.isdir(d):
        continue

    # --- Fix 1: VideoInput import in processing_eagle2_5_vl.py ---
    for fname in ["processing_eagle2_5_vl.py", "image_processing_eagle2_5_vl_fast.py"]:
        fpath = os.path.join(d, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            content = f.read()
        changed = False
        old_vi = "from transformers.video_utils import VideoInput"
        new_vi = (
            "try:\n"
            "    from transformers.video_utils import VideoInput\n"
            "except ImportError:\n"
            "    from transformers.image_utils import ImageInput as VideoInput  # compat"
        )
        if old_vi in content and "try:" not in content.split(old_vi)[0][-10:]:
            content = content.replace(old_vi, new_vi)
            changed = True

        if changed:
            with open(fpath, "w") as f:
                f.write(content)
            print(f"Patched VideoInput: {fpath}")
        else:
            print(f"VideoInput OK: {os.path.basename(fpath)}")

    # --- Fix 2: _prepare_image_like_inputs shim in fast processor ---
    # Must go BEFORE @add_start_docstrings decorator (not between decorator and class)
    fast_path = os.path.join(d, "image_processing_eagle2_5_vl_fast.py")
    if not os.path.exists(fast_path):
        continue
    with open(fast_path) as f:
        lines = f.readlines()

    SHIM_LINES = [
        "# _prepare_image_like_inputs compat shim\n",
        "if not hasattr(BaseImageProcessorFast, '_prepare_image_like_inputs'):\n",
        "    BaseImageProcessorFast._prepare_image_like_inputs = BaseImageProcessorFast._prepare_input_images\n",
    ]
    # First: remove any existing shim lines (may be mis-placed)
    cleaned = [l for l in lines if l.rstrip("\n") + "\n" not in SHIM_LINES]

    # Find the @add_start_docstrings line that precedes Eagle25VLImageProcessorFast
    insert_idx = None
    for i, l in enumerate(cleaned):
        if l.strip().startswith("@add_start_docstrings("):
            # Check if this decorator is for Eagle25VLImageProcessorFast
            for j in range(i + 1, min(i + 30, len(cleaned))):
                if "class Eagle25VLImageProcessorFast" in cleaned[j]:
                    insert_idx = i
                    break
            if insert_idx is not None:
                break

    if insert_idx is None:
        # Fallback: find class line directly
        for i, l in enumerate(cleaned):
            if "class Eagle25VLImageProcessorFast" in l:
                insert_idx = i
                break

    if insert_idx is not None:
        cleaned = cleaned[:insert_idx] + ["\n"] + SHIM_LINES + ["\n"] + cleaned[insert_idx:]
        with open(fast_path, "w") as f:
            f.writelines(cleaned)
        print(f"Patched _prepare_image_like_inputs: {fast_path}")
    else:
        print(f"WARN: Eagle25VLImageProcessorFast not found in {fast_path}")
EAGLE_PATCH
echo "Eagle2.5 patching done."

# Pre-download the dataset
echo "Pre-downloading Dex1 Sim dataset..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_LEROBOT_HOME']
snapshot_download('unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim', repo_type='dataset', local_dir=f'{cache}/unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim')
print('Download complete')
" || echo "WARNING: pre-download failed, training will attempt download on-the-fly"

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_groot_n15_dex1sim_blockstack_v5

accelerate launch \
  --multi_gpu --num_processes="$NUM_GPUS" --num_machines=1 --mixed_precision=no \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="$OUTPUT_DIR" \
  --policy.type=groot \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.embodiment_tag=unitree_g1 \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.lora_rank=16 \
  --policy.lora_alpha=32 \
  --policy.lora_dropout=0.05 \
  --policy.use_bf16=true \
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
  --wandb.notes="GR00T-N1.5 Dex1 Sim v5: 4GPU, reinject_dataset_stats fix, Eagle2.5 fast proc compat"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
