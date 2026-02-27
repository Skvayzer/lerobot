
import torch
from transformers import AutoConfig, AutoModel
from lerobot.policies.grootCoT.groot_n1 import QwenBackbone

def reproduce_nan():
    print("Initializing QwenBackbone...")
    # Matches configuration from policy
    backbone = QwenBackbone(
        tune_projector=True,
        select_layer=12,  # As per logs
        load_bf16=True,
        project_to_dim=2048, # Match user config
        attn_implementation="flash_attention_2",
        # LoRA Config as per logs (simplified)
        lora_config={
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        } 
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)
    
    # Input stats from log:
    # [GROOT][DEBUG] pixel_values stats: shape=(6256, 1176), min=-1.7890625, max=2.140625, mean=0.021240234375, dtype=torch.bfloat16
    # [GROOT][DEBUG] image_grid_thw: shape=(1, 4, 3), values=[[1, 34, 46], [1, 34, 46], [1, 34, 46], [1, 34, 46]]
    
    # Try to load real failing inputs
    import os
    if os.path.exists("debug_nan_input.pt"):
        print("Loading real failing inputs from debug_nan_input.pt...")
        # Since we use BatchFeature which is a custom class, we need weights_only=False or safe_globals
        vl_input = torch.load("debug_nan_input.pt", map_location=device, weights_only=False)
        # Ensure all tensors are on device and bf16 where appropriate
        for k, v in vl_input.items():
            if isinstance(v, torch.Tensor):
                vl_input[k] = v.to(device)
                if k == "eagle_pixel_values":
                     vl_input[k] = v.to(dtype=torch.bfloat16)

        pixel_values = vl_input["eagle_pixel_values"]
        image_grid_thw = vl_input.get("eagle_image_grid_thw")
        input_ids = vl_input.get("eagle_input_ids")
        attention_mask = vl_input.get("eagle_attention_mask")
        
        print(f"Loaded inputs: pixel_values={pixel_values.shape}, grid={image_grid_thw.shape if image_grid_thw is not None else 'None'}")
        if input_ids is not None:
             print(f"input_ids: {input_ids.shape}, min={input_ids.min()}, max={input_ids.max()}")
        if attention_mask is not None:
             print(f"attention_mask: {attention_mask.shape}, sum={attention_mask.sum()}")
    else:
        print("debug_nan_input.pt not found. Using random... (this probably won't reproduce it)")
        # ... logic for random generation (omitted for brevity, assume script falls back or we rely on file)
        return
    
    print("Running forward pass...")
    try:
        backbone.train() # Enable training mode (dropout etc)
        
        # We need to call forward_qwen directly or via forward
        # QwenBackbone.forward calls self.forward_qwen internally
        
        # But QwenBackbone.forward expects BatchFeature/dict with 'eagle_' prefix
        features, mask = backbone.forward_eagle(vl_input) # Assuming internal method name or similar?
        # Actually QwenBackbone logic seems to be inside forward_qwen method which is called by something?
        # Check groot_n1.py again. 
        # GrootCoT's QwenBackbone has forward_qwen?
        
        # Let's just use the public API if possible.
        # But wait, I need to see if I can call forward directly
        pass
    except Exception as e:
        print(f"Standard forward failed: {e}")

    # Let's inspect the backbone class methods briefly to call it correctly in script
    # It seems to have forward_qwen
    
    print("Calling forward_qwen directly...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        embeds, mask = backbone.forward_qwen(vl_input)
    
    print("Checking outputs...")
    if torch.isnan(embeds).any():
        print("!! NaNs detected in output !!")
    else:
        print("Output is clean.")

if __name__ == "__main__":
    reproduce_nan()
