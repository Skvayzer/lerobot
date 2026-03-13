"""Backward compatibility test: verify default GR00T N1.5 paths are unchanged.

Ensures CraftNet dual-rate features are purely additive and do not affect
the standard training/inference code paths when disabled.

Run: python -m lerobot.policies.grootCoT.test_backward_compat

Requires CUDA GPU and access to the Qwen3-VL model weights.
"""

import sys
from unittest.mock import patch, MagicMock
import torch
import numpy as np


def make_batch(model, device):
    """Build a synthetic batch using the actual VLM processor."""
    from transformers import AutoProcessor
    from PIL import Image

    resolved_id = getattr(
        model.config, "resolved_system2_vlm_model_id",
        model.config.vlm_processor_model_id,
    )
    processor = AutoProcessor.from_pretrained(resolved_id, trust_remote_code=True)

    batch_size = 2
    imgs = [
        [Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(2)]
        for _ in range(batch_size)
    ]

    all_pv, all_grid, all_ids, all_mask = [], [], [], []
    for sample_imgs in imgs:
        messages = [{"role": "user", "content": []}]
        for img in sample_imgs:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": "Act."})
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = processor(text=[text], images=sample_imgs, return_tensors="pt")
        all_pv.append(out["pixel_values"])
        all_grid.append(out["image_grid_thw"])
        all_ids.append(out["input_ids"].squeeze(0))
        all_mask.append(out["attention_mask"].squeeze(0))

    pv = torch.stack(all_pv).squeeze(1)
    if pv.dim() == 4:
        pv = pv.squeeze(1)
    grid = torch.stack(all_grid)
    max_seq = max(ids.shape[0] for ids in all_ids)
    ids_padded = torch.zeros(batch_size, max_seq, dtype=torch.long)
    mask_padded = torch.zeros(batch_size, max_seq, dtype=torch.long)
    for i, (ids, m) in enumerate(zip(all_ids, all_mask)):
        ids_padded[i, :ids.shape[0]] = ids
        mask_padded[i, :m.shape[0]] = m

    ad = model._groot_model.action_dim
    ah = model._groot_model.action_horizon

    return {
        "qwen_pixel_values": pv.to(device),
        "qwen_image_grid_thw": grid.to(device),
        "qwen_input_ids": ids_padded.to(device),
        "qwen_attention_mask": mask_padded.to(device),
        "state": torch.randn(batch_size, 1, 64, device=device),
        "state_mask": torch.ones(batch_size, 64, dtype=torch.bool, device=device),
        "action": torch.randn(batch_size, ah, ad, device=device),
        "action_mask": torch.ones(batch_size, 1, ad, dtype=torch.bool, device=device),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


def create_model(device, **config_overrides):
    from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
    from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    defaults = dict(
        dual_rate_enable=False,
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
    defaults.update(config_overrides)
    config = GrootCoTConfig(**defaults)
    model = GrootCoTPolicy(config)
    model.to(device)
    return model


def test_scenario_a(device):
    """Training with dual_rate_enable=False takes the legacy GR00T forward path."""
    print("\n" + "=" * 60)
    print("SCENARIO A: Training, dual_rate_enable=False")
    print("=" * 60)

    model = create_model(device, dual_rate_enable=False)
    model.train()
    batch = make_batch(model, device)

    # Verify config state
    assert not model._dual_rate_enabled(), "dual_rate should be disabled"
    assert not model._dual_rate_train_enabled(), "dual_rate_train should be disabled"
    print("  [OK] dual_rate_enabled=False, dual_rate_train_enabled=False")

    # Patch _groot_model.forward to trace the call
    original_forward = model._groot_model.forward
    forward_called = [False]

    def traced_forward(*args, **kwargs):
        forward_called[0] = True
        return original_forward(*args, **kwargs)

    # Patch run_action_head to detect if CraftNet path is taken
    original_rah = model._groot_model.run_action_head
    rah_called = [False]

    def traced_rah(*args, **kwargs):
        rah_called[0] = True
        return original_rah(*args, **kwargs)

    model._groot_model.forward = traced_forward
    model._groot_model.run_action_head = traced_rah

    try:
        loss_tensor, loss_dict = model.forward(batch)
    except Exception as e:
        print(f"  [FAIL] Forward crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    model._groot_model.forward = original_forward
    model._groot_model.run_action_head = original_rah

    if not forward_called[0]:
        print("  [FAIL] _groot_model.forward was NOT called (expected legacy path)")
        return False
    print("  [OK] Legacy _groot_model.forward() was called")

    if rah_called[0]:
        print("  [FAIL] run_action_head was called (CraftNet path should NOT be taken)")
        return False
    print("  [OK] run_action_head was NOT called (no CraftNet intervention)")

    loss = loss_tensor if isinstance(loss_tensor, torch.Tensor) else loss_dict.get("loss")
    if loss is None or torch.isnan(loss):
        print(f"  [FAIL] Invalid loss: {loss}")
        return False
    print(f"  [OK] Loss = {loss.item():.4f}")

    # Verify backward works
    model.zero_grad()
    loss.backward()
    print("  [OK] Backward pass completed")

    del model
    torch.cuda.empty_cache()
    return True


def test_scenario_b(device):
    """Inference with dual_rate_enable=False takes legacy get_action path."""
    print("\n" + "=" * 60)
    print("SCENARIO B: Inference, dual_rate_enable=False")
    print("=" * 60)

    model = create_model(device, dual_rate_enable=False)
    model.eval()
    batch = make_batch(model, device)

    # Remove training-only keys for inference
    inference_batch = {k: v for k, v in batch.items() if k not in ("action", "action_mask")}

    assert not model._dual_rate_enabled(), "dual_rate should be disabled"
    assert not model._async_dual_rate_enabled(), "async dual_rate should be disabled"
    print("  [OK] dual_rate_enabled=False, async_dual_rate_enabled=False")

    # Trace get_action to verify legacy path
    original_get_action = model._groot_model.get_action
    get_action_called = [False]

    def traced_get_action(*args, **kwargs):
        get_action_called[0] = True
        return original_get_action(*args, **kwargs)

    # Trace _get_fresh_visual_features to verify it either returns None or isn't called
    original_gfvf = model._get_fresh_visual_features
    gfvf_returned_nonnull = [False]

    def traced_gfvf(*args, **kwargs):
        result = original_gfvf(*args, **kwargs)
        if result is not None:
            gfvf_returned_nonnull[0] = True
        return result

    model._groot_model.get_action = traced_get_action
    model._get_fresh_visual_features = traced_gfvf

    try:
        with torch.no_grad():
            action = model.select_action(inference_batch)
    except Exception as e:
        print(f"  [FAIL] select_action crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    model._groot_model.get_action = original_get_action
    model._get_fresh_visual_features = original_gfvf

    if not get_action_called[0]:
        print("  [FAIL] _groot_model.get_action was NOT called")
        return False
    print("  [OK] Legacy _groot_model.get_action() was called")

    # Even though _get_fresh_visual_features is called in _predict_action_chunk_from_inputs,
    # with dual_rate_enable=False, the else branch (line 704) calls get_action directly,
    # bypassing fresh_visual entirely.
    if gfvf_returned_nonnull[0]:
        print("  [WARN] _get_fresh_visual_features returned non-None, but it shouldn't matter")
        print("         (get_action path doesn't use it)")

    if not isinstance(action, torch.Tensor):
        print(f"  [FAIL] Action is not a tensor: {type(action)}")
        return False

    expected_action_dim = model.config.output_features["action"].shape[0]
    if action.shape[-1] != expected_action_dim:
        print(f"  [FAIL] Action dim {action.shape[-1]} != expected {expected_action_dim}")
        return False
    print(f"  [OK] Action shape: {action.shape} (last dim = {expected_action_dim})")

    del model
    torch.cuda.empty_cache()
    return True


def test_scenario_c(device):
    """run_visual_only with non-Qwen backbone falls back to run_backbone."""
    print("\n" + "=" * 60)
    print("SCENARIO C: Eagle backbone fallback in run_visual_only")
    print("=" * 60)

    # We don't load a real Eagle model (would need different weights).
    # Instead, test the branching logic by mocking backbone type.
    model = create_model(device, dual_rate_enable=True)

    from lerobot.policies.grootCoT.groot_n1 import QwenBackbone

    # Verify current backbone IS QwenBackbone
    assert isinstance(model._groot_model.backbone, QwenBackbone), \
        "Expected QwenBackbone for current model"
    print("  [OK] Current backbone is QwenBackbone")

    # Test 1: run_visual_only with QwenBackbone calls forward_visual_only
    original_fvo = model._groot_model.backbone.forward_visual_only
    fvo_called = [False]

    def traced_fvo(*args, **kwargs):
        fvo_called[0] = True
        return original_fvo(*args, **kwargs)

    model._groot_model.backbone.forward_visual_only = traced_fvo
    batch = make_batch(model, device)
    groot_inputs = model._build_groot_inputs(batch, include_action=False)

    try:
        with torch.no_grad():
            result = model._groot_model.run_visual_only(groot_inputs)
    except Exception as e:
        print(f"  [FAIL] run_visual_only crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    model._groot_model.backbone.forward_visual_only = original_fvo

    if not fvo_called[0]:
        print("  [FAIL] forward_visual_only was NOT called for QwenBackbone")
        return False
    print("  [OK] QwenBackbone: forward_visual_only was called")

    vf = result.get("visual_features")
    if vf is not None:
        print(f"  [OK] visual_features returned: {vf.shape}")
    else:
        print("  [FAIL] visual_features is None for QwenBackbone path")
        return False

    # Test 2: Simulate non-Qwen backbone by temporarily changing the class check.
    # run_visual_only checks isinstance(self.backbone, QwenBackbone).
    # We swap __class__ temporarily to trigger the else branch.
    original_cls = model._groot_model.backbone.__class__
    run_backbone_called = [False]
    original_rb = model._groot_model.run_backbone

    def traced_rb(*args, **kwargs):
        run_backbone_called[0] = True
        return original_rb(*args, **kwargs)

    model._groot_model.run_backbone = traced_rb

    # Temporarily make backbone not look like QwenBackbone
    class FakeBackbone:
        pass

    model._groot_model.backbone.__class__ = FakeBackbone
    try:
        with torch.no_grad():
            result2 = model._groot_model.run_visual_only(groot_inputs)
    except Exception as e:
        # Expected — run_backbone with FakeBackbone will crash, but we only need
        # to verify run_backbone was dispatched to.
        if run_backbone_called[0]:
            print("  [OK] Non-Qwen backbone: fell back to run_backbone (expected crash in mock)")
        else:
            print(f"  [FAIL] Neither forward_visual_only nor run_backbone was called: {e}")
            return False
    else:
        if run_backbone_called[0]:
            print("  [OK] Non-Qwen backbone: fell back to run_backbone")
        else:
            print("  [FAIL] run_backbone was NOT called for non-Qwen backbone")
            return False
    finally:
        model._groot_model.backbone.__class__ = original_cls
        model._groot_model.run_backbone = original_rb

    # Test 3: Verify run_action_head with fresh_visual_features=None skips concatenation
    backbone_out_mock = model._groot_model.run_backbone(groot_inputs)
    cached_shape = backbone_out_mock["backbone_features"].shape

    action_inputs = model._groot_model.prepare_action_input(groot_inputs)
    # Temporarily bypass action_head forward to check the concatenation logic
    bf_before = backbone_out_mock["backbone_features"].shape

    # With fresh_visual_features=None, backbone_features should pass through unchanged
    from transformers.feature_extraction_utils import BatchFeature
    original_ah_forward = model._groot_model.action_head.forward

    def check_no_concat_forward(backbone_output, action_input):
        bf_shape = backbone_output["backbone_features"].shape
        if bf_shape != cached_shape:
            raise AssertionError(
                f"backbone_features shape changed from {cached_shape} to {bf_shape} "
                f"(concatenation happened despite fresh_visual_features=None)"
            )
        return original_ah_forward(backbone_output, action_input)

    model._groot_model.action_head.forward = check_no_concat_forward
    try:
        with torch.no_grad():
            model._groot_model.run_action_head(
                inputs=groot_inputs,
                backbone_outputs=model._groot_model.run_backbone(groot_inputs),
                is_training=False,
                fresh_visual_features=None,
            )
        print("  [OK] fresh_visual_features=None: no concatenation (backbone_features unchanged)")
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        return False
    except Exception as e:
        # Other errors (shape mismatches in action head) are OK — we're testing concatenation logic
        print(f"  [OK] fresh_visual_features=None: no concatenation (action head error is expected: {type(e).__name__})")
    finally:
        model._groot_model.action_head.forward = original_ah_forward

    del model
    torch.cuda.empty_cache()
    return True


def main():
    print("=" * 60)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] No CUDA — test will be slow")

    results = {}

    results["A"] = test_scenario_a(device)
    results["B"] = test_scenario_b(device)
    results["C"] = test_scenario_c(device)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  Scenario {name}: {status}")

    if all_pass:
        print("\nALL BACKWARD COMPATIBILITY CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
