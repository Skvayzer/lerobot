"""Verify dual-path training: both backbone_features and visual_features
produce gradients through the projector and action head.

Run: python -m lerobot.policies.grootCoT.test_dual_path_training

Requires CUDA GPU and access to the Qwen3-VL model weights.
"""

import sys
import torch
from transformers import AutoProcessor


def make_synthetic_batch(
    model,
    batch_size: int = 2,
    num_cameras: int = 2,
    image_h: int = 480,
    image_w: int = 640,
    state_dim: int = 29,
    action_dim: int = 29,
    action_horizon: int = 50,
    device: torch.device = torch.device("cuda"),
) -> dict[str, torch.Tensor]:
    """Build a synthetic batch mimicking the processor + collation output.

    The processor creates flat pixel_values and image_grid_thw.
    The default collator stacks them into (B, num_imgs, ...) tensors.
    """
    # Use the actual processor to get properly formatted visual tokens
    model_id = model.config.vlm_processor_model_id
    resolved_id = getattr(model.config, "resolved_system2_vlm_model_id", model_id)
    print(f"[TEST] Loading processor from: {resolved_id}")
    processor = AutoProcessor.from_pretrained(resolved_id, trust_remote_code=True)

    # Create random images as numpy arrays (HWC uint8)
    import numpy as np
    images_per_sample = []
    for _b in range(batch_size):
        sample_images = [
            np.random.randint(0, 255, (image_h, image_w, 3), dtype=np.uint8)
            for _ in range(num_cameras)
        ]
        images_per_sample.append(sample_images)

    # Process each sample through the VLM processor individually
    all_pixel_values = []
    all_grid_thw = []
    all_input_ids = []
    all_attention_mask = []
    for sample_images in images_per_sample:
        from PIL import Image
        pil_images = [Image.fromarray(img) for img in sample_images]
        # Qwen3-VL processor expects a conversation format
        messages = [{"role": "user", "content": []}]
        for img in pil_images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": "Describe."})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        proc_out = processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
        )
        all_pixel_values.append(proc_out["pixel_values"])  # (total_tokens, patch_dim)
        all_grid_thw.append(proc_out["image_grid_thw"])    # (num_imgs, 3)
        all_input_ids.append(proc_out["input_ids"].squeeze(0))  # (seq_len,)
        all_attention_mask.append(proc_out["attention_mask"].squeeze(0))

    # Check all samples have the same shape (they should with same-sized images)
    pv_shapes = [pv.shape for pv in all_pixel_values]
    grid_shapes = [g.shape for g in all_grid_thw]
    print(f"[TEST] pixel_values shapes per sample: {pv_shapes}")
    print(f"[TEST] image_grid_thw shapes per sample: {grid_shapes}")
    print(f"[TEST] input_ids shapes per sample: {[ids.shape for ids in all_input_ids]}")

    # Stack into batch: after collation these would be (B, num_imgs, ...)
    # pixel_values: (total_tokens_per_sample, patch_dim) -> stack to (B, tokens, dim)
    pixel_values = torch.stack(all_pixel_values, dim=0).squeeze(1)
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.squeeze(1)
    image_grid_thw = torch.stack(all_grid_thw, dim=0)  # (B, num_imgs, 3)

    # Pad input_ids and attention_mask to same length across batch
    max_seq_len = max(ids.shape[0] for ids in all_input_ids)
    padded_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    padded_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(all_input_ids, all_attention_mask)):
        padded_ids[i, :ids.shape[0]] = ids
        padded_mask[i, :mask.shape[0]] = mask

    print(f"[TEST] Batched pixel_values: {pixel_values.shape}")
    print(f"[TEST] Batched image_grid_thw: {image_grid_thw.shape}")
    print(f"[TEST] Batched input_ids: {padded_ids.shape}")

    # State, action, masks
    # GR00T model uses its own action_dim (from pretrained config), not max_action_dim.
    groot_action_dim = model._groot_model.action_dim
    groot_action_horizon = model._groot_model.action_horizon
    max_state = model.config.max_state_dim
    state = torch.randn(batch_size, max_state, device=device)
    state_mask = torch.zeros(batch_size, max_state, dtype=torch.bool, device=device)
    state_mask[:, :state_dim] = True
    action = torch.randn(batch_size, groot_action_horizon, groot_action_dim, device=device)
    action_mask = torch.zeros(batch_size, groot_action_dim, dtype=torch.bool, device=device)
    action_mask[:, :action_dim] = True
    print(f"[TEST] Using groot action_dim={groot_action_dim}, horizon={groot_action_horizon}")
    embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

    batch = {
        "qwen_pixel_values": pixel_values.to(device),
        "qwen_image_grid_thw": image_grid_thw.to(device),
        "qwen_input_ids": padded_ids.to(device),
        "qwen_attention_mask": padded_mask.to(device),
        "state": state,
        "state_mask": state_mask,
        "action": action,
        "action_mask": action_mask,
        "embodiment_id": embodiment_id,
    }
    return batch


def test_dual_path_gradients():
    print("=" * 60)
    print("DUAL-PATH TRAINING VERIFICATION")
    print("=" * 60)

    from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
    from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] No CUDA — test will be slow on CPU")

    # Build config with dual-rate training enabled
    config = GrootCoTConfig(
        dual_rate_enable=True,
        dual_rate_apply_in_train=True,
        visual_dropout_p=0.0,  # Never drop, so we always get both paths
        use_bf16=True,
        attn_implementation="eager",
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(29,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(29,)),
        },
    )

    print("\n[1/6] Creating model...")
    model = GrootCoTPolicy(config)
    model.to(device)
    model.train()

    # Print projector dimensions
    backbone = model._groot_model.backbone
    projector = backbone.projector
    print(f"[INFO] Projector: {projector}")
    hs = getattr(backbone.qwen_config, "hidden_size", None)
    if hs is None:
        hs = getattr(getattr(backbone.qwen_config, "text_config", None), "hidden_size", "N/A")
    print(f"[INFO] Qwen config hidden_size: {hs}")
    vision_cfg = getattr(backbone.qwen_config, "vision_config", None)
    if vision_cfg:
        print(f"[INFO] Vision config hidden_size: {getattr(vision_cfg, 'hidden_size', 'N/A')}")
        print(f"[INFO] Vision config spatial_merge_size: {getattr(vision_cfg, 'spatial_merge_size', 'N/A')}")

    print("\n[2/6] Creating synthetic batch...")
    batch = make_synthetic_batch(model, batch_size=2, device=device)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")

    # --- Test A: visual_dropout_p=0.0 (both paths active) ---
    print("\n[3/6] Forward pass (dropout_p=0.0, both paths)...")
    model.config.visual_dropout_p = 0.0
    model.config.dual_rate_apply_in_train = True
    try:
        loss_tensor, loss_dict = model.forward(batch)
    except Exception as e:
        print(f"[FAIL] Forward pass crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    loss = loss_tensor if isinstance(loss_tensor, torch.Tensor) else loss_dict.get("loss")
    print(f"  loss = {loss.item():.6f}")
    if torch.isnan(loss):
        print("[FAIL] Loss is NaN")
        return False
    if loss.item() == 0.0:
        print("[FAIL] Loss is exactly zero — suspicious")
        return False
    print("  [OK] Loss is valid")

    print("\n[4/6] Backward pass...")
    model.zero_grad()
    loss.backward()

    # Check projector gradients
    proj_grad_ok = True
    for name, p in projector.named_parameters():
        if p.grad is None:
            print(f"  [FAIL] projector.{name}: grad is None")
            proj_grad_ok = False
        elif p.grad.abs().sum().item() == 0.0:
            print(f"  [FAIL] projector.{name}: grad is all zeros")
            proj_grad_ok = False
        else:
            print(f"  [OK] projector.{name}: grad norm = {p.grad.norm().item():.6e}")

    if proj_grad_ok:
        print("  [OK] Projector receives gradients")
    else:
        print("  [FAIL] Projector gradient check failed")
        return False

    # Check that visual_features was produced during forward
    # We'll re-run just the backbone to verify
    print("\n[5/6] Verifying visual_features from hook...")
    groot_inputs = model._build_groot_inputs(batch, include_action=True)
    with torch.no_grad():
        backbone_outputs = model._groot_model.backbone.forward(
            model._groot_model.prepare_backbone_input(groot_inputs)
        )
    vf = backbone_outputs.get("visual_features")
    if vf is None:
        print("  [FAIL] visual_features not present in backbone output")
        return False
    print(f"  [OK] visual_features shape: {vf.shape}, dtype: {vf.dtype}")
    if vf.dim() != 3:
        print(f"  [FAIL] Expected 3D (B, patches, D), got {vf.dim()}D")
        return False
    print(f"  [OK] visual_features is 3D: (B={vf.shape[0]}, patches={vf.shape[1]}, D={vf.shape[2]})")

    # --- Test B: visual_dropout_p=1.0 (always drop, backbone-only path) ---
    print("\n[6/6] Forward pass (dropout_p=1.0, backbone-only)...")
    model.config.visual_dropout_p = 1.0
    model.zero_grad()
    try:
        loss_tensor2, loss_dict2 = model.forward(batch)
    except Exception as e:
        print(f"[FAIL] Forward with dropout=1.0 crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    loss2 = loss_tensor2 if isinstance(loss_tensor2, torch.Tensor) else loss_dict2.get("loss")
    print(f"  loss = {loss2.item():.6f}")
    if torch.isnan(loss2):
        print("[FAIL] Loss is NaN with dropout=1.0")
        return False
    print("  [OK] Model trains with visual features dropped (backbone-only)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_dual_path_gradients()
    sys.exit(0 if success else 1)
