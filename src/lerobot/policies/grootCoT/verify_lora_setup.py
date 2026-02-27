
import torch
from lerobot.policies.grootCoT.configuration_groot import GrootCoTConfig
from lerobot.policies.grootCoT.modeling_groot import GrootCoTPolicy

def verify_setup():
    print("Initializing GrootCoTConfig...")
    config = GrootCoTConfig(
        base_model_path="nvidia/GR00T-N1.5-3B",
        vlm_processor_model_id="Qwen/Qwen2-VL-2B-Instruct",
        train_vlm_projector_only=False,
        tune_llm=True,
        tune_visual=True,
        tune_projector=True,
        tune_diffusion_model=True,
        use_bf16=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        action_head_lora_rank=16,
        attn_implementation="flash_attention_2"
    )
    
    # Add dummy features to pass validation
    from lerobot.configs.types import PolicyFeature, FeatureType
    config.input_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(64,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
    }

    print("Creating GrootCoTPolicy...")
    # This triggers model loading
    policy = GrootCoTPolicy(config)
    
    model = policy._groot_model
    backbone = model.backbone
    
    print("\n=== Model Structure ===")
    print(f"Backbone type: {type(backbone)}")
    if hasattr(backbone.qwen_model, "peft_config"):
        print("Backbone is wrapped with PEFT.")
        backbone.qwen_model.print_trainable_parameters()
    else:
        print("WARNING: Backbone is NOT wrapped with PEFT!")

    print("\n=== Precision Check ===")
    param_dtypes = {}
    for name, param in backbone.named_parameters():
        dtype = str(param.dtype)
        if dtype not in param_dtypes:
            param_dtypes[dtype] = 0
        param_dtypes[dtype] += 1
    print(f"Backbone parameter dtypes distribution: {param_dtypes}")

    print("\n=== Truncation Check ===")
    if hasattr(backbone.qwen_model, "model") and hasattr(backbone.qwen_model.model, "layers"):
        layers = backbone.qwen_model.model.layers
        print(f"Number of Qwen layers: {len(layers)}")
        print(f"Selected layer index: {backbone.select_layer}")
        if len(layers) <= backbone.select_layer:
            print(f"SUCCESS: Model truncated correctly (layers <= {backbone.select_layer})")
        else:
             # Wait, if select_layer is 1-based index or 0-based?
             # select_layer is used as index: hidden_states[select_layer]
             # If select_layer is 16, typically it means 16th layer output.
             # Truncation logic pop(-1) while len > select_layer means we keep [0..select_layer-1] indices?
             # If len == select_layer, max index is select_layer-1.
             # So hidden_states[select_layer] would be out of bounds if we use standard indexing?
             # But transformers output hidden_states includes embeddings at 0.
             # So hidden_states has len(layers)+1 elements.
             # If we have N layers, we have N+1 hidden states.
             # If we want hidden_state[K], we need layer K-1 to exist.
             print(f"WARNING: Model might not be truncated (is {len(layers)} <= {backbone.select_layer}?)")
    else:
        print("Could not find layers attribute to verify truncation.")

    print("\n=== Memory Check ===")
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"CUDA Memory Allocated: {mem_alloc:.2f} GB")
    print(f"CUDA Memory Reserved:  {mem_reserved:.2f} GB")

if __name__ == "__main__":
    verify_setup()
