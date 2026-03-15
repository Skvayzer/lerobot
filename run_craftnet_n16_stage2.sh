#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH --job-name=cn16_s2
#SBATCH --exclude=auh7-1b-gpu-215
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_craftnet_n16_stage2_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_craftnet_n16_stage2_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

# =====================================================
# UPDATE THIS to actual Stage 1 checkpoint path
# =====================================================
STAGE1_CHECKPOINT="outputs/train/YYYYMMDD_HHMMSS_craftnet_n16_stage1/checkpoints/005000/pretrained_model"

if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CHECKPOINT"
    exit 1
fi

WEIGHTS_PATH="$LEROBOT_DIR/src/lerobot/policies/grootCoT/action_head_n16/n16_action_head_weights.pt"

echo "=========================================="
echo "CraftNet N1.6 Stage 2: DiT Fine-tuning + Relative Actions"
echo "  Stage 1 checkpoint: $STAGE1_CHECKPOINT"
echo "  Training: DiT + VLLN + projectors + top 4 LLM layers"
echo "  Relative actions: enabled"
echo "  LR: 1e-4 -> 1e-6 over 20K steps"
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

CACHE_ROOT=/tmp/$USER/rocm_cache_${SLURM_JOB_ID}
mkdir -p "$CACHE_ROOT"/{miopen_db,miopen_cache,torch_kernels,xdg_cache}
chmod -R u+rwX "$CACHE_ROOT"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen_cache"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1

NUM_GPUS=4
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
git pull origin unitree-features || echo "WARNING: git pull failed"

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_craftnet_n16_stage2

accelerate launch \
  --multi_gpu --num_processes="$NUM_GPUS" --num_machines=1 --mixed_precision=no \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir="$OUTPUT_DIR" \
  \
  --policy.type=groot_cot \
  --policy.pretrained_path="$STAGE1_CHECKPOINT" \
  --policy.embodiment_tag=unitree_g1 \
  --policy.push_to_hub=false \
  \
  --policy.action_head_version=n16 \
  --policy.chunk_size=50 \
  --policy.n_action_steps=50 \
  \
  --policy.train_vlm_projector_only=false \
  --policy.training_stage=manual \
  --policy.tune_llm=true \
  --policy.tune_visual=false \
  --policy.tune_vlm_projector=true \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.tune_vlln=true \
  --policy.tune_top_llm_layers=4 \
  \
  --policy.use_relative_actions=true \
  --policy.flare_enable=false \
  \
  --policy.use_bf16=false \
  --policy.dual_rate_enable=false \
  --policy.dual_rate_apply_in_train=false \
  --policy.visual_dropout_p=0.0 \
  \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_num_decay_steps=20000 \
  --policy.scheduler_decay_lr_ratio=0.01 \
  --policy.optimizer_grad_clip_norm=1.0 \
  --policy.optimizer_weight_decay=1e-5 \
  \
  --dataset.repo_id=unitreerobotics/G1_Dex3_BlockStacking_Dataset \
  --dataset.root="$HF_LEROBOT_HOME/unitreerobotics/G1_Dex3_BlockStacking_Dataset" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --tolerance_s=5e-4 \
  --dataset.tolerance_s=5e-4 \
  \
  --batch_size=4 \
  --num_workers=2 \
  --steps=20000 \
  --save_freq=2000 \
  --keep_last_n_checkpoints=5 \
  \
  --wandb.enable=true \
  --wandb.project=CraftNet_N16 \
  --wandb.entity=skvayzer \
  --wandb.disable_artifact=true \
  --wandb.notes="CraftNet N1.6 Stage 2: DiT finetune + relative actions, LR=1e-4->1e-6, 20K steps"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
