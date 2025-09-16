#!/usr/bin/env python3
"""
Simple test script to verify Qwen SWIFT implementation works correctly.
This script tests basic model loading and SWIFT functionality.
"""

import torch
from transformers import AutoTokenizer
from model.swift.modeling_qwen import Qwen3ForCausalLM


def test_qwen_swift_basic():
    """Test basic Qwen SWIFT model functionality."""
    print("Testing Qwen SWIFT implementation...")
    
    # Test that we can import the model
    print("Successfully imported Qwen3ForCausalLM")
    
    # Test SWIFT methods
    print("Testing SWIFT methods...")
    
    # Create a dummy config for testing
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    
    config = Qwen3Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512
    )
    
    # Create model instance
    model = Qwen3ForCausalLM(config)
    print("Successfully created Qwen3ForCausalLM instance")
    
    # Test SWIFT methods
    # Test set_skip_layers
    model.set_skip_layers([1], [1])
    attn_skip, mlp_skip = model.get_skip_layers()
    assert attn_skip == [1], f"Expected [1], got {attn_skip}"
    assert mlp_skip == [1], f"Expected [1], got {mlp_skip}"
    print("set_skip_layers and get_skip_layers work correctly")
    
    # Test self_draft context manager
    with model.self_draft(enabled=True):
        pass
    print("self_draft context manager works correctly")
    
    # Test add_bitfit
    model.add_bitfit()
    print("add_bitfit works correctly")
    
    # Test forward pass with dummy input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Test forward without SWIFT parameters
    outputs = model(input_ids)
    assert outputs.logits is not None
    print("Forward pass works correctly")
    
    # Test forward with SWIFT parameters
    draft_attn_skip_mask = torch.zeros(config.num_hidden_layers, dtype=torch.bool)
    draft_mlp_skip_mask = torch.zeros(config.num_hidden_layers, dtype=torch.bool)
    
    outputs = model(
        input_ids,
        draft_attn_skip_mask=draft_attn_skip_mask,
        draft_mlp_skip_mask=draft_mlp_skip_mask
    )
    assert outputs.logits is not None
    print("Forward pass with SWIFT parameters works correctly")
    
    print("\nAll tests passed! Qwen SWIFT implementation is working correctly.")
    return True


if __name__ == "__main__":
    try:
        test_qwen_swift_basic()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)