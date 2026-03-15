#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH --job-name=craftnet_stage1
#SBATCH --exclude=auh7-1b-gpu-215
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_craftnet_stage1_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_craftnet_stage1_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "CraftNet Stage 1: Projector Alignment"
echo "  - VLM projector + action head projector train"
echo "  - Qwen backbone frozen, DiT frozen"
echo "  - No dual-rate, no visual dropout"
echo "  - High LR (1e-3) for fast projector convergence"
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
export HIPBLAS_OP_DTYPE_FP32=1

CURRENT_TF=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "Current transformers: $CURRENT_TF"

# CraftNet needs transformers >= 4.57 for Qwen3-VL
python -c "from transformers import Qwen3VLForConditionalGeneration" 2>/dev/null || {
    echo "Installing transformers >= 4.57 for Qwen3-VL support..."
    pip install 'transformers==4.57.1' 2>&1 | tail -3
    echo "transformers now: $(python -c 'import transformers; print(transformers.__version__)')"
}

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

# Pre-download dataset
echo "Pre-downloading BlockStacking dataset..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_LEROBOT_HOME']
snapshot_download('unitreerobotics/G1_Dex3_BlockStacking_Dataset', repo_type='dataset', local_dir=f'{cache}/unitreerobotics/G1_Dex3_BlockStacking_Dataset')
print('Download complete')
" || echo "WARNING: pre-download failed"

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_craftnet_blockstack_stage1

accelerate launch \
  --multi_gpu --num_processes="$NUM_GPUS" --num_machines=1 --mixed_precision=no \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="$OUTPUT_DIR" \
  \
  --policy.type=groot_cot \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.embodiment_tag=unitree_g1 \
  --policy.push_to_hub=false \
  \
  --policy.train_vlm_projector_only=true \
  --policy.training_stage=manual \
  \
  --policy.use_bf16=false \
  --policy.dual_rate_enable=false \
  --policy.dual_rate_apply_in_train=false \
  --policy.visual_dropout_p=0.0 \
  \
  --policy.optimizer_lr=1e-3 \
  --policy.scheduler_num_decay_steps=0 \
  --policy.scheduler_decay_lr_ratio=0.1 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.optimizer_weight_decay=1e-4 \
  \
  --dataset.repo_id=unitreerobotics/G1_Dex3_BlockStacking_Dataset \
  --dataset.root="$HF_LEROBOT_HOME/unitreerobotics/G1_Dex3_BlockStacking_Dataset" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=false \
  --tolerance_s=5e-4 \
  --dataset.tolerance_s=5e-4 \
  \
  --batch_size=8 \
  --num_workers=2 \
  --steps=5000 \
  --save_freq=1000 \
  --keep_last_n_checkpoints=5 \
  \
  --wandb.enable=true \
  --wandb.project=CraftNet_Baselines \
  --wandb.entity=skvayzer \
  --wandb.disable_artifact=true \
  --wandb.notes="CraftNet Stage 1: projector alignment, LR=1e-3, DiT frozen, 5K steps, 8GPU"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
