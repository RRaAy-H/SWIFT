#!/usr/bin/env python3
"""
Analyze SWIFT memory usage with 4 GPUs
"""

def calculate_kvcache_memory():
    """Calculate memory usage of KVCache with 4 GPUs"""
    print("🔍 SWIFT KVCache Memory Analysis with 4 GPUs")
    print("=" * 60)

    # Qwen3-4B-Instruct configuration
    config = {
        'num_hidden_layers': 32,
        'num_key_value_heads': 8,
        'max_position_embeddings': 32768,
        'head_dim': 128,  # Our calculated head_dim
        'num_gpus': 4
    }

    batch_size = 1
    dtype_size = 2  # float16 = 2 bytes

    print(f"Model Configuration:")
    print(f"  - Layers: {config['num_hidden_layers']}")
    print(f"  - KV Heads: {config['num_key_value_heads']}")
    print(f"  - Max Seq Length: {config['max_position_embeddings']}")
    print(f"  - Head Dimension: {config['head_dim']}")
    print(f"  - GPUs: {config['num_gpus']}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Data Type: float16 ({dtype_size} bytes)")
    print()

    # Calculate memory per layer (key + value)
    elements_per_layer = (
        2 *  # key + value
        batch_size *
        config['num_key_value_heads'] *
        config['max_position_embeddings'] *
        config['head_dim']
    )

    memory_per_layer = elements_per_layer * dtype_size
    print(f"Memory per layer: {memory_per_layer:,} bytes = {memory_per_layer / (1024**3):.2f} GB")

    # Total memory for all layers (distributed across GPUs)
    total_elements = elements_per_layer * config['num_hidden_layers']
    total_memory = total_elements * dtype_size
    print(f"Total KVCache memory: {total_memory:,} bytes = {total_memory / (1024**3):.2f} GB")

    # Memory per GPU (distributed)
    layers_per_gpu = config['num_hidden_layers'] // config['num_gpus']
    memory_per_gpu = memory_per_layer * layers_per_gpu
    print(f"Memory per GPU (distributed): {memory_per_gpu:,} bytes = {memory_per_gpu / (1024**3):.2f} GB")

    print()
    print("🚨 SWIFT Draft Generation Memory Issue:")
    print("-" * 40)

    # The problem: swift_draft creates a FULL CLONE of ALL KVCache data
    clone_memory = total_memory  # Full clone of all KVCache data
    print(f"Draft clone memory: {clone_memory:,} bytes = {clone_memory / (1024**3):.2f} GB")

    # This clone might be created on a single GPU instead of being distributed
    print(f"If clone created on single GPU: {clone_memory / (1024**3):.2f} GB")
    print("This is likely what's causing the 11 GB allocation attempt!")

    print()
    print("💡 Analysis:")
    print(f"  - Original KVCache: {total_memory / (1024**3):.2f} GB (distributed across 4 GPUs)")
    print(f"  - Draft clone: {clone_memory / (1024**3):.2f} GB (possibly on single GPU)")
    print(f"  - Total peak usage: {(total_memory + clone_memory) / (1024**3):.2f} GB")

    if clone_memory / (1024**3) > 10:
        print("  - ⚠️  Clone exceeds 10 GB - explains the OOM error!")

    return total_memory, clone_memory

def analyze_error():
    """Analyze the specific error message"""
    print("\n🔍 Error Message Analysis:")
    print("-" * 30)
    print("Error: 'Tried to allocate 11.00 GiB. GPU 1 has ... 9.88 GiB is free'")
    print()
    print("This suggests:")
    print("1. SWIFT tried to allocate 11 GB on GPU 1 specifically")
    print("2. GPU 1 only has 9.88 GB free")
    print("3. The allocation is for the draft KVCache clone")
    print("4. The clone is NOT being distributed across GPUs")

def propose_solutions():
    """Propose solutions for the memory issue"""
    print("\n💡 Proposed Solutions:")
    print("=" * 30)

    print("1. 🎯 Fix clone distribution:")
    print("   - Ensure draft clones are created on the same GPU as original tensors")
    print("   - Modify swift_draft to respect GPU placement")

    print("\n2. 🔧 Reduce memory footprint:")
    print("   - Use smaller max_position_embeddings")
    print("   - Implement lazy cloning (only clone what's needed)")

    print("\n3. ⚡ Optimize cloning strategy:")
    print("   - Use tensor views instead of full clones where possible")
    print("   - Implement incremental cloning")

if __name__ == "__main__":
    total_mem, clone_mem = calculate_kvcache_memory()
    analyze_error()
    propose_solutions()