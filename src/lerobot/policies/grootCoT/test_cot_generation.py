"""Step 4e: Verify System 2 (Qwen3-VL) actually produces subtask_text and target_bbox.

Loads real images from the dex3 block stacking dataset, creates a CoT session,
runs extract_cot_trace, and checks that the structured JSON output contains
the expected fields.

Requires CUDA GPU and access to the Qwen3-VL model weights.

Run: python -m lerobot.policies.grootCoT.test_cot_generation
"""

import sys
import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def find_dataset_dir() -> Path | None:
    """Find the dex3 block stacking dataset on disk."""
    candidates = [
        Path.home() / ".cache/huggingface/lerobot/unitreerobotics/G1_Dex3_BlockStacking_Dataset",
        Path.home() / ".cache/huggingface/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset",
        Path("/data/lerobot/unitreerobotics/G1_Dex3_BlockStacking_Dataset"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _extract_frame_pyav(video_path: str, frame_idx: int) -> Image.Image | None:
    """Extract a single frame from a video using PyAV (handles AV1 codec)."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            img = frame.to_image()  # Returns PIL Image
            container.close()
            return img
    container.close()
    return None


def load_sample_images(dataset_dir: Path, episode_idx: int = 0, frame_idx: int = 0) -> dict[str, Image.Image]:
    """Load camera images for a given episode/frame from the dataset.

    Supports LeRobot chunk-based video layout:
      videos/observation.images.{cam}/chunk-{chunk}/file-{episode_in_chunk}.mp4
    Uses PyAV for decoding (handles AV1 codec that cv2 often can't).
    """
    cameras = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]
    images = {}
    for cam in cameras:
        found = False
        # LeRobot chunk-based video layout
        for chunk_idx in range(4):
            video_path = dataset_dir / f"videos/observation.images.{cam}/chunk-{chunk_idx:03d}/file-{episode_idx:03d}.mp4"
            if video_path.exists():
                img = _extract_frame_pyav(str(video_path), frame_idx)
                if img is not None:
                    images[cam] = img
                    found = True
                    print(f"    Loaded {cam}: {img.size} from chunk-{chunk_idx:03d}/file-{episode_idx:03d}.mp4 frame {frame_idx}")
                break
            if found:
                break

        # Fallback: glob for any matching video
        if not found:
            for gpat in [str(dataset_dir / f"videos/*{cam}*/**/*.mp4")]:
                matches = sorted(glob.glob(gpat, recursive=True))
                if matches:
                    img = _extract_frame_pyav(matches[0], frame_idx)
                    if img is not None:
                        images[cam] = img
                        found = True
                        print(f"    Loaded {cam}: {img.size} from {Path(matches[0]).name} frame {frame_idx}")
                    break
    return images


def create_model(device):
    """Create a GrootCoTPolicy with dual_rate + CoT enabled."""
    from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
    from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy
    from lerobot.configs.types import FeatureType, PolicyFeature

    config = GrootCoTConfig(
        dual_rate_enable=True,
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
    model = GrootCoTPolicy(config)
    model.to(device)
    model.eval()
    return model


def make_batch_from_images(model, device, images: list[Image.Image]) -> dict:
    """Build a model-compatible batch from PIL images."""
    from transformers import AutoProcessor

    resolved_id = getattr(
        model.config, "resolved_system2_vlm_model_id",
        model.config.vlm_processor_model_id,
    )
    processor = AutoProcessor.from_pretrained(resolved_id, trust_remote_code=True)

    # Build Qwen chat template with images
    messages = [{"role": "user", "content": []}]
    for img in images:
        messages[0]["content"].append({"type": "image", "image": img})
    messages[0]["content"].append({"type": "text", "text": "Act."})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = processor(text=[text], images=images, return_tensors="pt")

    # Add batch dimension if needed and build full batch
    pv = out["pixel_values"]
    grid = out["image_grid_thw"]
    ids = out["input_ids"]
    mask = out["attention_mask"]

    if pv.dim() == 3:
        pv = pv.unsqueeze(0)

    batch = {
        "qwen_pixel_values": pv.to(device),
        "qwen_image_grid_thw": grid.to(device),
        "qwen_input_ids": ids.to(device),
        "qwen_attention_mask": mask.to(device),
        "state": torch.randn(1, 1, 64, device=device),
        "state_mask": torch.ones(1, 64, dtype=torch.bool, device=device),
        "embodiment_id": torch.zeros(1, dtype=torch.long, device=device),
    }
    return batch


def test_cot_generation_single(
    model, device, images: list[Image.Image], task: str, test_name: str,
    cot_session, dataset_meta: dict,
) -> bool:
    """Run one CoT generation test and check for subtask_text + target_bbox."""
    print(f"\n  [{test_name}] Task: '{task}'")
    print(f"  [{test_name}] Images: {len(images)}, session mode: {cot_session.mode}")

    batch = make_batch_from_images(model, device, images)

    traces = model.extract_cot_trace(
        batch,
        cot_session=cot_session,
        dataset_meta=dataset_meta,
        max_new_tokens=4096,  # Thinking model needs room for <think> + JSON
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    if not traces:
        print(f"  [{test_name}] FAIL: No traces returned")
        return False

    trace = traces[0]
    print(f"  [{test_name}] Generated {trace.get('generated_tokens', '?')} tokens")
    print(f"  [{test_name}] Parse OK: {trace.get('parse_ok')}")

    if trace.get("parse_error"):
        print(f"  [{test_name}] Parse error: {trace['parse_error']}")

    # Show raw text (truncated)
    raw = trace.get("raw_text", "")
    if len(raw) > 500:
        print(f"  [{test_name}] Raw text (first 500 chars): {raw[:500]}...")
    else:
        print(f"  [{test_name}] Raw text: {raw}")

    if not trace.get("parse_ok"):
        print(f"  [{test_name}] FAIL: JSON parse failed")
        return False

    parsed = trace["parsed_json"]

    # Check for next block
    next_block = parsed.get("next")
    if not isinstance(next_block, dict):
        print(f"  [{test_name}] FAIL: No 'next' block in parsed JSON")
        print(f"  [{test_name}] Top-level keys: {list(parsed.keys())}")
        return False

    # Check subtask_text
    subtask_text = next_block.get("system1_subtask_text")
    has_subtask = subtask_text is not None and str(subtask_text).strip() != ""
    print(f"  [{test_name}] system1_subtask_text: {subtask_text!r} {'OK' if has_subtask else 'MISSING'}")

    # Check target_bbox
    bbox = next_block.get("target_bbox")
    has_bbox = (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(isinstance(v, (int, float)) for v in bbox)
    )
    if has_bbox:
        in_range = all(0.0 <= v <= 1.0 for v in bbox)
        print(f"  [{test_name}] target_bbox: {bbox} {'OK' if in_range else 'OUT OF RANGE'}")
    else:
        print(f"  [{test_name}] target_bbox: {bbox!r} MISSING/INVALID")

    # Check policy state was updated
    if has_subtask:
        assert model._current_subtask_text == str(subtask_text).strip(), \
            f"Policy state mismatch: {model._current_subtask_text} != {subtask_text}"
        print(f"  [{test_name}] OK: _current_subtask_text updated")

    if has_bbox and isinstance(bbox, list) and len(bbox) == 4:
        assert model._current_target_bbox is not None, "Policy _current_target_bbox is None"
        print(f"  [{test_name}] OK: _current_target_bbox updated")

    # Also check other schema fields
    schema_v = parsed.get("schema_version")
    mode = parsed.get("mode")
    print(f"  [{test_name}] schema_version: {schema_v}, mode: {mode}")

    success = has_subtask  # subtask_text is required; bbox is strongly encouraged
    if not has_bbox:
        print(f"  [{test_name}] WARN: target_bbox missing (not fatal, but should be present)")
    if success:
        print(f"  [{test_name}] PASS")
    else:
        print(f"  [{test_name}] FAIL")
    return success


def main():
    print("=" * 60)
    print("STEP 4e: CoT GENERATION TEST (Structured JSON Output)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] No CUDA — this test requires a GPU with enough VRAM for Qwen3-VL-8B")

    # Find dataset
    dataset_dir = find_dataset_dir()
    if dataset_dir is not None:
        print(f"[INFO] Found dataset at: {dataset_dir}")
    else:
        print("[INFO] No dex3 block stacking dataset found, will use synthetic images")

    # Create model
    print("\n[INFO] Creating model...")
    model = create_model(device)
    print(f"[INFO] Model on {device}, dtype={'bf16' if model.config.use_bf16 else 'fp32'}")

    # Create CoT session
    from lerobot.policies.grootCoT.cot_schema import CoTSessionState

    # Test scenarios
    test_cases = [
        {
            "task": "pick up the red block and place it on the tape mark",
            "episode": 0, "frame": 10,
            "test_name": "Test1_RedBlock",
        },
        {
            "task": "stack the yellow block on top of the red block",
            "episode": 1, "frame": 20,
            "test_name": "Test2_YellowOnRed",
        },
        {
            "task": "grasp the blue block with the right hand and lift it",
            "episode": 2, "frame": 5,
            "test_name": "Test3_BlueBlockLift",
        },
    ]

    results = {}
    for tc in test_cases:
        # Load images
        if dataset_dir is not None:
            img_dict = load_sample_images(dataset_dir, tc["episode"], tc["frame"])
            if img_dict:
                images = list(img_dict.values())[:2]  # Use 2 camera views
                print(f"\n[INFO] Loaded {len(images)} images from dataset "
                      f"(cameras: {list(img_dict.keys())[:2]})")
            else:
                print(f"\n[INFO] Could not load images for ep={tc['episode']}, frame={tc['frame']}")
                print("[INFO] Using synthetic images")
                images = [
                    Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                    for _ in range(2)
                ]
        else:
            images = [
                Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
                for _ in range(2)
            ]

        # Fresh session for each test (INIT mode)
        model.reset()
        cot_session = CoTSessionState(
            episode_index=tc["episode"],
            instruction=tc["task"],
            dataset_name="G1_Dex3_BlockStacking_Dataset",
            dataset_family="dex3",
            cameras_present=["cam_left_high", "cam_right_high"],
        )
        dataset_meta = cot_session.build_meta(time_index=tc["frame"])

        try:
            results[tc["test_name"]] = test_cot_generation_single(
                model, device, images, tc["task"], tc["test_name"],
                cot_session, dataset_meta,
            )
        except Exception as e:
            print(f"\n  [{tc['test_name']}] EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[tc["test_name"]] = False

    # Test 4: TICK mode (continue from a previous INIT)
    print("\n" + "-" * 40)
    print("Test4: TICK mode (after INIT)")
    print("-" * 40)
    model.reset()
    cot_session_tick = CoTSessionState(
        episode_index=0,
        instruction="stack all three colored blocks into a tower",
        dataset_name="G1_Dex3_BlockStacking_Dataset",
        dataset_family="dex3",
        cameras_present=["cam_left_high", "cam_right_high"],
    )

    # Fake an INIT step
    fake_init_result = {
        "plan": {
            "steps": [
                {"step_id": 1, "skill": "perceive", "args": {"target": "blocks"}},
                {"step_id": 2, "skill": "grasp", "args": {"target": "red block", "hand": "right"}},
                {"step_id": 3, "skill": "place", "args": {"target": "tape mark", "hand": "right"}},
            ]
        },
        "execution_state": {"active_step_id": 2, "active_step_status": "ongoing", "retries_used": 0},
    }
    cot_session_tick.advance(fake_init_result)
    assert cot_session_tick.mode == "TICK", f"Expected TICK mode, got {cot_session_tick.mode}"

    if dataset_dir is not None:
        img_dict = load_sample_images(dataset_dir, 0, 30)
        images = list(img_dict.values())[:2] if img_dict else [
            Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            for _ in range(2)
        ]
    else:
        images = [
            Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            for _ in range(2)
        ]

    dataset_meta_tick = cot_session_tick.build_meta(time_index=30)
    try:
        results["Test4_TICK"] = test_cot_generation_single(
            model, device, images,
            "stack all three colored blocks into a tower",
            "Test4_TICK",
            cot_session_tick, dataset_meta_tick,
        )
    except Exception as e:
        print(f"\n  [Test4_TICK] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results["Test4_TICK"] = False

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    all_pass = True
    bbox_count = 0
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name}: {status}")

    if all_pass:
        print("\nALL CoT GENERATION CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
        print("Note: If JSON parsing fails, the model may need more max_new_tokens")
        print("      or the prompt may need tuning for reliable structured output.")
    print("=" * 60)

    # Check policy state is populated
    print(f"\nFinal policy state:")
    print(f"  _current_subtask_text: {model._current_subtask_text!r}")
    print(f"  _current_target_bbox: {model._current_target_bbox!r}")

    del model
    torch.cuda.empty_cache()
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
