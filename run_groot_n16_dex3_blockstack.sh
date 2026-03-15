#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=2-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=512G
#SBATCH --job-name=n16_dex3_bs
#SBATCH --exclude=auh7-1b-gpu-215
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_n16_dex3_blockstack_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_n16_dex3_blockstack_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "GR00T N1.6 Baseline — Dex3 BlockStacking"
echo "  Model: nvidia/GR00T-N1.6-3B"
echo "  Dataset: G1_Dex3_BlockStacking_Dataset"
echo "  8x MI210, single node"
echo "Node: $(hostname) | Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

source /vast/users/chenyuan.chen/miniconda3/bin/activate unitree_lerobot_amd_n16

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/vast/users/chenyuan.chen/.cache/huggingface
export HF_LEROBOT_HOME=/vast/users/chenyuan.chen/.cache/huggingface/lerobot
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HIPBLAS_OP_DTYPE_FP32=1

echo "Python: $(which python)"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "gr00t: $(python -c 'import gr00t; print(gr00t.__version__)' 2>/dev/null || echo 'not installed')"

# Patch Eagle3_VL for ROCm (flash_attn assertions)
GROOT_MODULES_DIR=/vast/users/chenyuan.chen/Isaac-GR00T/gr00t/model/modules
HF_EAGLE_CACHE=/vast/users/chenyuan.chen/.cache/huggingface/modules/transformers_modules/Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2
python3 - <<'PYEOF'
import os, re

TARGETS = [
    os.environ.get("GROOT_MODULES_DIR", "") + "/nvidia/Eagle-Block2A-2B-v2/modeling_eagle3_vl.py",
    os.environ.get("HF_EAGLE_CACHE", "") + "/modeling_eagle3_vl.py",
]

ASSERT_PATTERN = re.compile(
    r'assert \(\s*config\.text_config\._attn_implementation == "flash_attention_2"\s*\)'
    r',\s*f"(Qwen[23]) must use flash_attention_2 but got \{config\.text_config\._attn_implementation\}"',
    re.DOTALL,
)
ASSERT_REPLACEMENT = r'pass  # ROCm compat: relaxed flash_attention_2 assertion for \1'

for path in TARGETS:
    if not os.path.exists(path):
        continue
    with open(path) as f:
        content = f.read()
    patched = ASSERT_PATTERN.sub(ASSERT_REPLACEMENT, content)
    patched = patched.replace(
        'config.vision_config._attn_implementation = "flash_attention_2"',
        'config.vision_config._attn_implementation = "eager"',
    )
    if patched != content:
        with open(path, "w") as f:
            f.write(patched)
        print(f"Patched: {path}")
    else:
        print(f"Already patched: {path}")
PYEOF

CACHE_ROOT=/tmp/$USER/rocm_cache_${SLURM_JOB_ID}
mkdir -p "$CACHE_ROOT"/{miopen_db,miopen_cache,torch_kernels,xdg_cache}
chmod -R u+rwX "$CACHE_ROOT"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen_cache"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1

NUM_GPUS=8
echo "GPUs: $NUM_GPUS | Conda: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

git pull origin unitree-features || echo "WARNING: git pull failed, continuing"

# Pre-download dataset
echo "Pre-downloading BlockStacking dataset..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_LEROBOT_HOME']
snapshot_download('unitreerobotics/G1_Dex3_BlockStacking_Dataset', repo_type='dataset', local_dir=f'{cache}/unitreerobotics/G1_Dex3_BlockStacking_Dataset')
print('Download complete')
" || echo "WARNING: pre-download failed"

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_groot_n16_dex3_blockstack

accelerate launch \
  --multi_gpu --num_processes=$NUM_GPUS --num_machines=1 --mixed_precision=no \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="$OUTPUT_DIR" \
  \
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
  --policy.use_bf16=false \
  --policy.use_flash_attention=false \
  --policy.push_to_hub=false \
  \
  --dataset.repo_id=unitreerobotics/G1_Dex3_BlockStacking_Dataset \
  --dataset.root="$HF_LEROBOT_HOME/unitreerobotics/G1_Dex3_BlockStacking_Dataset" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --tolerance_s=5e-4 \
  --dataset.tolerance_s=5e-4 \
  \
  --batch_size=8 \
  --num_workers=4 \
  --epochs=5 \
  --save_freq=2000 \
  --keep_last_n_checkpoints=3 \
  \
  --wandb.enable=true \
  --wandb.project=G1_Groot_Baselines \
  --wandb.entity=skvayzer \
  --wandb.disable_artifact=true \
  --wandb.notes="GR00T N1.6 baseline, Dex3 BlockStacking, 8xMI210, FP32"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
