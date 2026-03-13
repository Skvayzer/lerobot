"""End-to-end test: System 2 sub-task state flows through to System 1 conditioning.

Verifies:
1. extract_subtask_fields parses subtask_text + bbox from CoT JSON
2. _update_subtask_from_traces updates policy state
3. get_subtask_overrides returns the correct state
4. Processor reads current_subtask_text and overrides language
5. Grounded reference frame renders bbox when present

Run: python -m lerobot.policies.grootCoT.test_subtask_pipeline
"""

import sys


def test_schema_extraction():
    """Step 3a: extract_subtask_fields correctly parses CoT JSON."""
    print("\n[1/5] Testing extract_subtask_fields...")
    from lerobot.policies.grootCoT.cot_schema import extract_subtask_fields

    # Full fields present
    parsed = {
        "next": {
            "system1_subtask_text": "grasp the red block with right hand",
            "target_bbox": [0.1, 0.2, 0.5, 0.6],
            "decision": "continue",
            "expected_horizon_steps": 30,
        }
    }
    text, bbox = extract_subtask_fields(parsed)
    assert text == "grasp the red block with right hand", f"Expected subtask text, got {text}"
    assert bbox == [0.1, 0.2, 0.5, 0.6], f"Expected bbox, got {bbox}"
    print("  [OK] Full fields extracted")

    # Missing fields (backward compat)
    text2, bbox2 = extract_subtask_fields({})
    assert text2 is None and bbox2 is None
    print("  [OK] Empty JSON returns None, None")

    # Partial: only subtask_text
    text3, bbox3 = extract_subtask_fields({"next": {"system1_subtask_text": "pick up block"}})
    assert text3 == "pick up block" and bbox3 is None
    print("  [OK] Partial fields (text only)")

    # Invalid bbox (wrong length)
    text4, bbox4 = extract_subtask_fields({"next": {"target_bbox": [0.1, 0.2]}})
    assert text4 is None and bbox4 is None
    print("  [OK] Invalid bbox rejected")

    # Empty string subtask
    text5, bbox5 = extract_subtask_fields({"next": {"system1_subtask_text": "  "}})
    assert text5 is None
    print("  [OK] Whitespace-only subtask treated as None")

    return True


def test_policy_state_update():
    """Steps 3b-3c: Policy state updates from CoT traces."""
    print("\n[2/5] Testing policy sub-task state management...")
    from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy
    from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    config = GrootCoTConfig(
        dual_rate_enable=True,
        use_bf16=True,
        attn_implementation="eager",
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(29,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(29,)),
        },
    )
    model = GrootCoTPolicy(config)

    # After reset, all sub-task state should be None/0
    assert model._current_subtask_text is None
    assert model._current_target_bbox is None
    assert model._subtask_index == 0
    print("  [OK] Initial state is clean after reset()")

    # Simulate a successful CoT trace
    fake_traces = [{
        "parse_ok": True,
        "parsed_json": {
            "next": {
                "system1_subtask_text": "reach for the yellow block",
                "target_bbox": [0.3, 0.4, 0.7, 0.8],
                "decision": "continue",
            }
        },
    }]
    model._update_subtask_from_traces(fake_traces)
    assert model._current_subtask_text == "reach for the yellow block"
    assert model._current_target_bbox == [0.3, 0.4, 0.7, 0.8]
    print("  [OK] _update_subtask_from_traces sets state correctly")

    # Failed parse should not overwrite
    model._update_subtask_from_traces([{"parse_ok": False}])
    assert model._current_subtask_text == "reach for the yellow block"
    print("  [OK] Failed parse does not overwrite state")

    # get_subtask_overrides returns current state
    overrides = model.get_subtask_overrides()
    assert overrides["current_subtask_text"] == "reach for the yellow block"
    assert overrides["target_bbox"] == [0.3, 0.4, 0.7, 0.8]
    print("  [OK] get_subtask_overrides returns correct dict")

    # Reset clears state
    model.reset()
    assert model._current_subtask_text is None
    assert model._current_target_bbox is None
    assert model._subtask_index == 0
    overrides2 = model.get_subtask_overrides()
    assert len(overrides2) == 0
    print("  [OK] reset() clears all sub-task state")

    del model
    return True


def test_processor_language_override():
    """Step 3d: Processor reads current_subtask_text from complementary data."""
    print("\n[3/5] Testing processor language override...")
    import torch
    import numpy as np
    from lerobot.policies.grootCoT.processor_groot import GrootPackInputsStep

    # Create a minimal processor step
    step = GrootPackInputsStep(
        max_state_dim=64,
        max_action_dim=32,
        action_horizon=16,
        video_height=480,
        video_width=640,
        camera_names=["front"],
    )

    # Build a minimal transition dict
    from lerobot.processors.preprocessor import TransitionKey
    transition = {
        TransitionKey.OBSERVATION: {
            "observation.images.front": torch.rand(1, 3, 480, 640),
            "observation.state": torch.randn(1, 29),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "assemble the windmill",
        },
    }

    # Without sub-task override
    result1 = step(dict(transition))
    comp1 = result1.get(TransitionKey.COMPLEMENTARY_DATA, {})
    lang1 = comp1.get("language", "")
    assert "assemble the windmill" in lang1, f"Expected original task in lang, got: {lang1}"
    print(f"  [OK] Without override: lang starts with original task")

    # With sub-task override
    transition2 = {
        TransitionKey.OBSERVATION: {
            "observation.images.front": torch.rand(1, 3, 480, 640),
            "observation.state": torch.randn(1, 29),
        },
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": "assemble the windmill",
            "current_subtask_text": "grasp the shaft with right hand",
        },
    }
    result2 = step(dict(transition2))
    comp2 = result2.get(TransitionKey.COMPLEMENTARY_DATA, {})
    lang2 = comp2.get("language", "")
    assert "grasp the shaft with right hand" in lang2, f"Expected subtask in lang, got: {lang2}"
    assert "assemble the windmill" not in lang2, f"Original task should be replaced, got: {lang2}"
    print(f"  [OK] With override: lang uses sub-task text")
    print(f"       lang = '{lang2[:60]}...'")

    return True


def test_grounded_reference_frame():
    """Step 3d+: Grounded reference frame renders bbox onto video."""
    print("\n[4/5] Testing grounded reference frame rendering...")
    import numpy as np
    from lerobot.policies.grootCoT.grounding_utils import render_grounded_frame

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = [0.25, 0.25, 0.75, 0.75]
    result = render_grounded_frame(frame, bbox, color=(255, 0, 0), thickness=3)

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    # Check that some pixels are red (bbox was drawn)
    red_pixels = (result[:, :, 0] == 255) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
    assert red_pixels.any(), "No red pixels found — bbox not rendered"
    print(f"  [OK] Bounding box rendered: {red_pixels.sum()} red pixels")

    return True


def test_cot_prompt_fields():
    """Step 3a: Verify target_bbox appears in INIT and TICK prompts."""
    print("\n[5/5] Testing CoT prompt field references...")
    from lerobot.policies.grootCoT.cot_schema import build_init_prompt, build_tick_prompt, SYSTEM_PROMPT

    meta = {
        "dataset_family": "dex3",
        "dataset_name": "test_dataset",
        "instruction": "stack the blocks",
        "time_index": 0,
    }

    _, init_user = build_init_prompt(meta)
    assert "target_bbox" in init_user, "target_bbox missing from INIT prompt"
    print("  [OK] INIT prompt mentions target_bbox")

    _, tick_user = build_tick_prompt(meta, last_plan=None, exec_state=None)
    assert "target_bbox" in tick_user, "target_bbox missing from TICK prompt"
    print("  [OK] TICK prompt mentions target_bbox")

    assert "target_bbox" in SYSTEM_PROMPT, "target_bbox missing from SYSTEM_PROMPT"
    print("  [OK] SYSTEM_PROMPT mentions target_bbox")

    return True


def main():
    print("=" * 60)
    print("SYSTEM 2 → SYSTEM 1 INTENT PIPELINE TEST")
    print("=" * 60)

    results = {}
    results["1_schema"] = test_schema_extraction()
    results["2_policy_state"] = test_policy_state_update()
    results["3_processor_lang"] = test_processor_language_override()
    results["4_grounded_frame"] = test_grounded_reference_frame()
    results["5_prompt_fields"] = test_cot_prompt_fields()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")

    if all_pass:
        print("\nALL SUBTASK PIPELINE CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
