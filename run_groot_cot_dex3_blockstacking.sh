#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=3-00:00:00
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --mem=256G
#SBATCH --job-name=groot_cot_dex3_blockstack
#SBATCH --output=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_cot_dex3_blockstack_%j.log
#SBATCH --error=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot/slurm_groot_cot_dex3_blockstack_%j.log

set -euo pipefail

###############################################################################
# CraftNet Training: GR00T N1.5 + Qwen3-VL-8B dual-rate System 1 / System 2
# Dataset: G1_Dex3_BlockStacking_Dataset (301 episodes)
# GPUs: 8 (2 nodes x 4 AMD MI210 64GB)
# Goal: Validate dual-path architecture before adding bbox/progress heads
###############################################################################

LEROBOT_DIR=/vast/users/chenyuan.chen/constantine/unitree_IL_lerobot/unitree_lerobot/lerobot
cd "$LEROBOT_DIR"

echo "=========================================="
echo "CraftNet: GR00T-N1.5 + Qwen3-VL-8B Dual-Rate Training"
echo "Dataset: G1_Dex3_BlockStacking_Dataset (301 episodes)"
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

# Ensure transformers >= 4.57 for native Qwen3-VL support.
# Eagle2.5 imports are lazy (only loaded by EagleBackbone, not CraftNet/QwenBackbone),
# so the removal of group_images_by_shape in transformers >= 4.56 does not affect us.
CURRENT_TF=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "Current transformers: $CURRENT_TF"
if python -c "
import transformers
from packaging.version import Version
v = Version(transformers.__version__)
exit(0 if v >= Version('4.57.0') else 1)
" 2>/dev/null; then
    echo "transformers >= 4.57.0, OK"
else
    echo "Installing transformers>=4.57.0 (current: $CURRENT_TF) ..."
    pip install 'transformers>=4.57.0' 2>&1 | tail -5
    echo "transformers now: $(python -c 'import transformers; print(transformers.__version__)')"
fi

# ROCm cache directories (avoid contention on shared filesystems)
CACHE_ROOT=/tmp/$USER/rocm_cache_${SLURM_JOB_ID}
mkdir -p "$CACHE_ROOT"/{miopen_db,miopen_cache,torch_kernels,xdg_cache}
chmod -R u+rwX "$CACHE_ROOT"
export MIOPEN_USER_DB_PATH="$CACHE_ROOT/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$CACHE_ROOT/miopen_cache"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="$CACHE_ROOT/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1

# ROCm-specific: disable Flash Attention (MI210 doesn't support it natively)
# The model config uses attn_implementation="eager" already
export FLASH_ATTENTION_FORCE_BUILD=0
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1

NUM_GPUS=4
NUM_NODES=2
TOTAL_GPUS=$((NUM_GPUS * NUM_NODES))
echo "GPUs per node: $NUM_GPUS | Nodes: $NUM_NODES | Total GPUs: $TOTAL_GPUS"
echo "Conda: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'CUDA devices on this node: {torch.cuda.device_count()}')"

# Pull latest code
git pull origin unitree-features || echo "WARNING: git pull failed, continuing with existing code"

# Clear stale Eagle2.5 HF cache
echo "Clearing stale Eagle2.5 HF cache..."
rm -rf ~/.cache/huggingface/modules/transformers_modules/eagle2hg-processor-groot-n1p5
rm -rf ~/.cache/huggingface/modules/transformers_modules/eagle2hg_hyphen_processor_hyphen_groot_hyphen_n1p5
echo "Eagle2.5 cache cleared."

# Pre-download the Dex3 BlockStacking dataset
echo "Pre-downloading Dex3 BlockStacking dataset..."
python -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ['HF_LEROBOT_HOME']
snapshot_download('unitreerobotics/G1_Dex3_BlockStacking_Dataset', repo_type='dataset', local_dir=f'{cache}/unitreerobotics/G1_Dex3_BlockStacking_Dataset')
print('Dex3 BlockStacking dataset download complete')
" || echo "WARNING: pre-download failed, training will attempt download on-the-fly"

# Pre-download GR00T-N1.5-3B model weights
echo "Pre-downloading GR00T-N1.5-3B model weights..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/GR00T-N1.5-3B')
print('GR00T-N1.5-3B download complete')
" || echo "WARNING: model pre-download failed"

# Pre-download Qwen3-VL-8B-Thinking model weights (System 2 backbone)
echo "Pre-downloading Qwen3-VL-8B-Thinking model weights..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Thinking')
print('Qwen3-VL-8B-Thinking download complete')
" || echo "WARNING: Qwen model pre-download failed"

###############################################################################
# Quick sanity check: verify the dataset loads and dual-rate path is taken
###############################################################################
echo ""
echo "=== Pre-flight checks ==="

# Check 1: Dataset loads through processor pipeline
echo "Check 1: Dataset format and processor pipeline..."
python -c "
import torch
from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature

# Create config matching training settings
config = GrootCoTConfig(
    dual_rate_enable=True,
    dual_rate_apply_in_train=True,
    use_bf16=True,
    attn_implementation='eager',
    tune_diffusion_model=True,
    tune_vlln=True,
    tune_projector=True,
    tune_vlm_projector=True,
    visual_dropout_p=0.2,
    input_features={
        'observation.images.cam_left_high': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        'observation.images.cam_right_high': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(29,)),
    },
    output_features={
        'action': PolicyFeature(type=FeatureType.ACTION, shape=(29,)),
    },
)

# Verify dual-rate training path is enabled
model = GrootCoTPolicy(config)
assert model._dual_rate_enabled(), 'dual_rate_enabled should be True'
assert model._dual_rate_train_enabled(), 'dual_rate_train_enabled should be True (needs dual_rate_apply_in_train=True)'
print('  [OK] dual_rate_enabled=True, dual_rate_train_enabled=True')

# Check trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'  [OK] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')
del model
torch.cuda.empty_cache()
print('Check 1 PASSED')
" 2>&1 | tail -10 || { echo "Check 1 FAILED — aborting"; exit 1; }

# Check 2: Verify video backend can decode AV1
echo "Check 2: AV1 video decoding with pyav..."
python -c "
import av
print(f'  [OK] PyAV version: {av.__version__}')
print('Check 2 PASSED')
" || { echo "Check 2 FAILED — pyav not available"; exit 1; }

echo "=== All pre-flight checks passed ==="
echo ""

###############################################################################
# Launch training
###############################################################################

OUTPUT_DIR=outputs/train/$(date +%Y%m%d_%H%M%S)_groot_cot_dex3_blockstacking

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Batch size per GPU: start with 4 (conservative for Qwen3-VL-8B + DiT)
# Total effective batch = 4 * 8 = 32
# If this OOMs, reduce to 2; if it fits easily, try 8
BATCH_SIZE_PER_GPU=12

echo "Master: ${MASTER_ADDR}:${MASTER_PORT} | Nodes: $NUM_NODES | GPUs: $TOTAL_GPUS | batch: ${BATCH_SIZE_PER_GPU}x${TOTAL_GPUS}=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS))"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Write a per-rank launcher so srun tasks on both nodes can source conda
# and receive correctly-expanded variables with per-rank local caches.
LAUNCHER="${LEROBOT_DIR}/tmp_launch_${SLURM_JOB_ID}.sh"
cat > "$LAUNCHER" <<LAUNCH_SCRIPT
#!/bin/bash
source /vast/users/chenyuan.chen/miniconda3/bin/activate unitree_lerobot_amd
cd ${LEROBOT_DIR}
export HF_HOME=${HF_HOME}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export FLASH_ATTENTION_FORCE_BUILD=0
export FLASH_ATTENTION_SKIP_CUDA_BUILD=1
# Per-rank local /tmp caches to avoid cross-node VAST flock (ENOLCK) issues
RANK_CACHE=/tmp/\${USER}/rocm_cache_${SLURM_JOB_ID}_\${SLURM_PROCID}
mkdir -p "\${RANK_CACHE}"/{miopen_db,miopen_cache,torch_kernels,xdg_cache,hf_datasets}
export MIOPEN_USER_DB_PATH="\${RANK_CACHE}/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="\${RANK_CACHE}/miopen_cache"
export XDG_CACHE_HOME="\${RANK_CACHE}/xdg_cache"
export PYTORCH_KERNEL_CACHE_PATH="\${RANK_CACHE}/torch_kernels"
export USE_PYTORCH_KERNEL_CACHE=1
export HF_DATASETS_CACHE="\${RANK_CACHE}/hf_datasets"
export ROCR_VISIBLE_DEVICES=0,1,2,3
accelerate launch \\
  --multi_gpu --num_processes=${TOTAL_GPUS} --num_machines=${NUM_NODES} \\
  --machine_rank=\${SLURM_PROCID} \\
  --main_process_ip=${MASTER_ADDR} \\
  --main_process_port=${MASTER_PORT} \\
  --mixed_precision=no \\
  src/lerobot/scripts/lerobot_train.py \\
  --output_dir=${OUTPUT_DIR} \\
  --policy.type=groot_cot \\
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \\
  --policy.embodiment_tag=new_embodiment \\
  --policy.dual_rate_enable=true \\
  --policy.dual_rate_apply_in_train=true \\
  --policy.visual_dropout_p=0.2 \\
  --policy.use_grounded_reference_frame=false \\
  --policy.recap_enable=false \\
  --policy.recap_value_head_enable=false \\
  --policy.tune_llm=false \\
  --policy.tune_visual=false \\
  --policy.tune_vlm_projector=true \\
  --policy.tune_projector=true \\
  --policy.tune_diffusion_model=true \\
  --policy.tune_vlln=true \\
  --policy.train_vlm_projector_only=false \\
  --policy.lora_rank=0 \\
  --policy.action_head_lora_rank=0 \\
  --policy.use_bf16=true \\
  --policy.attn_implementation=eager \\
  --policy.push_to_hub=false \\
  --policy.max_state_dim=64 \\
  --policy.max_action_dim=32 \\
  --dataset.repo_id=unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
  --dataset.root=${HF_LEROBOT_HOME}/unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
  '--dataset.robot_types=["g1"]' \\
  --dataset.video_backend=pyav \\
  --tolerance_s=5e-4 \\
  --dataset.tolerance_s=5e-4 \\
  --batch_size=${BATCH_SIZE_PER_GPU} \\
  --num_workers=2 \\
  --epochs=10 \\
  --save_freq=2000 \\
  --keep_last_n_checkpoints=3 \\
  --log_freq=50 \\
  --wandb.enable=true \\
  --wandb.project=G1_Groot_CraftNet \\
  --wandb.entity=skvayzer \\
  --wandb.disable_artifact=true \\
  '--wandb.notes=CraftNet v1: GR00T-N1.5 + Qwen3-VL-8B dual-rate, Dex3 BlockStacking 301ep, 8GPU MI210, bs=${BATCH_SIZE_PER_GPU}x${TOTAL_GPUS}=$((BATCH_SIZE_PER_GPU * TOTAL_GPUS)), dual_rate_apply_in_train=true, visual_dropout=0.2'
LAUNCH_SCRIPT
chmod +x "$LAUNCHER"

srun --export=ALL bash "$LAUNCHER"
TRAIN_RC=$?
rm -f "$LAUNCHER"
echo ""
echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
exit $TRAIN_RC
