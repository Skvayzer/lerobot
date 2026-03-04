# RECAP-Style Learning From Experience

This extension adds three opt-in components for GR00T-Qwen:

1. Value head on System-2 (Qwen) embeddings (`policy.recap_value_head_enable=true`).
2. Experience labeling utilities (reward/return/bin/advantage/indicator sidecar).
3. Indicator-conditioned policy fine-tuning with optional CFG steering at inference.

## 1) Label experience

```bash
python -m lerobot.data.label_experience \
  --input /path/to/rollouts.parquet \
  --output /path/to/rollouts_labeled.parquet \
  --c_fail 50 \
  --n_step 50 \
  --target_pos_rate 0.30 \
  --value_vmin -1.0 \
  --value_vmax 0.0 \
  --value_bins 201
```

Expected labeled columns include:
- `reward`
- `return`
- `value_target_bin`
- `advantage`
- `indicator`

## 2) Train value head

```bash
python -m lerobot.training.train_value \
  --config_path /path/to/train_config.json \
  --labels_path /path/to/rollouts_labeled.parquet \
  --output_dir outputs/value_head_run \
  --steps 10000
```

Notes:
- This trains the distributional value head attached to Qwen/System-2.
- The labels table must include `index` and `value_target_bin`.

## 3) Fine-tune policy with indicator conditioning

Use normal training entrypoint, enabling RECAP flags:

```bash
python src/lerobot/scripts/lerobot_train.py \
  --config_path=/path/to/train_config.json \
  --policy.recap_enable=true \
  --policy.recap_labels_path=/path/to/rollouts_labeled.parquet \
  --policy.recap_adv_indicator_dropout_p=0.3 \
  --policy.recap_value_head_enable=true
```

Or use wrapper:

```bash
python -m lerobot.training.train_policy_recap \
  --config_path=/path/to/train_config.json \
  --policy.recap_labels_path=/path/to/rollouts_labeled.parquet
```

## 4) Optional CFG steering at inference

Enable:
- `policy.recap_adv_indicator_use_cfg=true`
- `policy.recap_cfg_scale=1.5` (or >1.0)

Behavior:
- conditional branch uses `I_t=1`
- unconditional branch uses `I_t=NULL` (default `-1`)
- guided action prediction uses `uncond + scale * (cond - uncond)`

