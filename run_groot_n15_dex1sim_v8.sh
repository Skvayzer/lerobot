#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH --job-name=groot_n15_dex1sim_v8
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n15_dex1sim_v8_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n15_dex1sim_v8_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "GR00T N1.5 Dex1 Sim BlockStacking v8 (4 GPU, no LoRA)"
echo "Fixes vs v7:"
echo "  1. FP32 instead of BF16 (MI210 BF16 matmul precision issues)"
echo "  2. LR schedule spans full training (was exhausted at step 10K/77K)"
echo "  3. 20K steps (NVIDIA recommended) instead of 77K overfitting"
echo "  4. Grad clip 1.0 (HF Trainer default) instead of 10.0"
echo "  5. Image augmentation enabled"
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

# Eagle2.5 fast image processor requires transformers >= 4.55
CURRENT_TF=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "Current transformers: $CURRENT_TF"
if [ "$CURRENT_TF" = "4.55.0" ]; then
    echo "transformers 4.55.0, OK"
else
    echo "Installing transformers==4.55.0 (current: $CURRENT_TF) ..."
    pip install 'transformers==4.55.0' 2>&1 | tail -5
    echo "transformers now: $(python -c 'import transformers; print(transformers.__version__)')"
fi

CACHE_ROOT=/tmp/$USER/rocm_cache_${SLURM_JOB_ID}
mkdir -p "$CACHE_ROOT"/{miopen_db,miopen_cache,torch_kernels,xdg_cache}
chmod -R u+rwX "$CACHE_ROOT"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen_cache"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1

# Force FP32 on MI210 — BF16 matmuls produce incorrect gradients on gfx90a/ROCm 7.0
export HIPBLAS_OP_DTYPE_FP32=1

NUM_GPUS=4
echo "GPUs: $NUM_GPUS | Conda: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

git pull origin unitree-features || echo "WARNING: git pull failed, continuing"

# Clean up stale Eagle2.5 HF cache
echo "Clearing stale Eagle2.5 HF cache (will re-download fresh)..."
rm -rf ~/.cache/huggingface/modules/transformers_modules/eagle2hg-processor-groot-n1p5
rm -rf ~/.cache/huggingface/modules/transformers_modules/eagle2hg_hyphen_processor_hyphen_groot_hyphen_n1p5
echo "Eagle2.5 cache cleared."

# Pre-download the dataset
echo "Pre-downloading Dex1 Sim dataset..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_LEROBOT_HOME']
snapshot_download('unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim', repo_type='dataset', local_dir=f'{cache}/unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim')
print('Download complete')
" || echo "WARNING: pre-download failed, training will attempt download on-the-fly"

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_groot_n15_dex1sim_blockstack_v8

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
  --policy.use_bf16=false \
  --policy.scheduler_num_decay_steps=0 \
  --policy.scheduler_decay_lr_ratio=0.01 \
  --policy.push_to_hub=false \
  --dataset.repo_id=unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim \
  --dataset.root="$HF_LEROBOT_HOME/unitreerobotics/G1_Dex1_StackRygBlock_Dataset_Sim" \
  --dataset.robot_types='["g1"]' \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --tolerance_s=5e-4 \
  --dataset.tolerance_s=5e-4 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --batch_size=8 \
  --num_workers=2 \
  --steps=20000 \
  --save_freq=2000 \
  --keep_last_n_checkpoints=5 \
  --wandb.enable=true \
  --wandb.project=G1_Groot_Baselines \
  --wandb.entity=skvayzer \
  --wandb.disable_artifact=true \
  --wandb.notes="GR00T-N1.5 Dex1 Sim v8: FP32, LR spans full 20K, grad_clip=1.0, img aug ON"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
