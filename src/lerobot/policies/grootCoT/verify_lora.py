
import sys
import unittest
from unittest.mock import MagicMock, patch
import torch
from torch import nn

# Adjust path to find lerobot
import os
sys.path.append(os.getcwd())

# Import real libraries first
import transformers
from peft import PeftModel, LoraConfig

# Now import the classes to test
from lerobot.policies.grootCoT.groot_n1 import QwenBackbone, GR00TN15
from lerobot.policies.groot.action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig

class TestGrootLoRA(unittest.TestCase):
    def setUp(self):
        # Create patches
        self.patcher1 = patch('transformers.AutoConfig.from_pretrained')
        self.patcher2 = patch('transformers.Qwen3VLForConditionalGeneration.from_pretrained')
        self.patcher3 = patch('transformers.AutoModel.from_pretrained')
        
        self.mock_config_load = self.patcher1.start()
        self.mock_qwen_load = self.patcher2.start()
        self.mock_automodel_load = self.patcher3.start()


        # Setup Config mock
        self.mock_config_obj = MagicMock()
        self.mock_config_obj.hidden_size = 1024
        self.mock_config_obj.num_hidden_layers = 12
        # Need text_config for QwenBackbone checks
        self.mock_config_obj.text_config.hidden_size = 1024
        self.mock_config_obj.text_config.num_hidden_layers = 12
        self.mock_config_load.return_value = self.mock_config_obj
        
        # Setup real nn.Module structure to satisfy Peft
        class SimpleMockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        "self_attn": nn.ModuleDict({
                            "q_proj": nn.Linear(10, 10),
                            "k_proj": nn.Linear(10, 10),
                            "v_proj": nn.Linear(10, 10),
                        })
                    })
                ])
                # For ActionHead testing (DiT structure simulation if needed, or just generic)
                # DiT usually has transformer_blocks...
                # But FlowmatchingActionHead wraps "self.model" which IS the DiT.
                
            def forward(self, *args, **kwargs):
                return None

        self.mock_qwen_model_real = SimpleMockModel()
        self.mock_qwen_load.return_value = self.mock_qwen_model_real
        self.mock_automodel_load.return_value = self.mock_qwen_model_real

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()


    def test_qwen_backbone_lora(self):
        print("\nTesting QwenBackbone LoRA...")
        # Must specify target_modules for Mock model as Peft can't infer architecture
        lora_config = {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q_proj"]}
        
        # Instantiate QwenBackbone with LoRA config
        backbone = QwenBackbone(
            model_id="dummy",
            tune_vlm=True,
            lora_config=lora_config
        )
        
        # Verify get_peft_model was called
        # Since we can't easily check isinstance(backbone.qwen_model, PeftModel) because get_peft_model returns a real PeftModel wrapping our Mock, 
        # and PeftModel requires a real model structure usually.
        # But wait, QwenBackbone code calls: self.qwen_model = get_peft_model(self.qwen_model, peft_config)
        # So backbone.qwen_model SHOULD be a PeftModel.
        
        self.assertTrue(isinstance(backbone.qwen_model, PeftModel), "Qwen model should be wrapped in PeftModel")
        print("Success: QwenBackbone wrapped with PeftModel")

    def test_action_head_lora(self):
        print("\nTesting FlowmatchingActionHead LoRA...")
        config = FlowmatchingActionHeadConfig(
            input_embedding_dim=64,
            backbone_embedding_dim=64,
            hidden_size=64,
            action_dim=10,
            action_horizon=5,
            max_state_dim=64,
            tune_diffusion_model=True,
            use_vlln=False,
            add_pos_embed=False,
            # Config for DiT to be lightweight
            diffusion_model_cfg={
                "num_layers": 1,
                "attention_head_dim": 32,
                "num_attention_heads": 2,
                "output_dim": 10
            }
        )
        
        head = FlowmatchingActionHead(config)
        
        # Initial check: model is DiT (not PeftModel)
        self.assertFalse(isinstance(head.model, PeftModel))
        
        # Apply LoRA
        lora_config = {"r": 4, "lora_alpha": 8, "target_modules": ["to_q", "to_v"]} # Target modules for DiT Attention
        head.set_trainable_parameters(tune_projector=True, tune_diffusion_model=True, lora_config=lora_config)
        
        # Verify
        self.assertTrue(isinstance(head.model, PeftModel), "ActionHead DiT should be wrapped in PeftModel")
        print("Success: FlowmatchingActionHead wrapped with PeftModel")

if __name__ == "__main__":
    unittest.main()
