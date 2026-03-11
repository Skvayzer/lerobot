#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=512G
#SBATCH --job-name=groot_n16_dex1sim
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n16_dex1sim_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_n16_dex1sim_%j.log

set -euo pipefail

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "GR00T N1.6 Dex1 Sim BlockStacking"
echo "Model: nvidia/GR00T-N1.6-3B (32-layer AlternateVLDiT)"
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

# Install gr00t package into training env if not already installed
echo "Checking gr00t package installation..."
if ! python -c "import gr00t" 2>/dev/null; then
    echo "Installing gr00t package from /home/cosmos/Isaac-GR00T ..."
    pip install -e /home/cosmos/Isaac-GR00T
    echo "gr00t installed."
else
    echo "gr00t already installed: $(python -c 'import gr00t; print(gr00t.__version__)')"
fi

CACHE_ROOT=/tmp/$USER/rocm_cache_${SLURM_JOB_ID}
mkdir -p "$CACHE_ROOT"/{miopen_db,miopen_cache,torch_kernels,xdg_cache}
chmod -R u+rwX "$CACHE_ROOT"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen_cache"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=${SLURM_GPUS_ON_NODE:-8}
echo "GPUs: $NUM_GPUS | Conda: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

git pull origin unitree-features || echo "WARNING: git pull failed, continuing"

# Pre-download the dataset to avoid DataLoader worker download stalls
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

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_groot_n16_dex1sim_blockstack

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
  --batch_size=32 \
  --num_workers=4 \
  --epochs=5 \
  --save_freq=2000 \
  --keep_last_n_checkpoints=2 \
  --wandb.enable=true \
  --wandb.project=G1_Groot_Baselines \
  --wandb.entity=skvayzer \
  --wandb.disable_artifact=true \
  --wandb.notes="GR00T N1.6 Dex1 Sim: 32-layer AlternateVLDiT, max_state/action_dim=29"

echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
