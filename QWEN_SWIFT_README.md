# Qwen SWIFT Implementation

## Key Features
### Qwen3-Specific Adaptations
- **Query/Key Normalization**: Preserves Qwen3's q_norm and k_norm in attention
- **Sliding Window Attention**: Supports Qwen3's sliding window attention patterns
- **Layer Types**: Handles different attention types (full_attention, sliding_attention)
- **RoPE Embeddings**: Maintains Qwen3's rotary position embeddings

## Usage

### Quick Start with Evaluation Script

The easiest way to run both baseline and SWIFT evaluations:

```bash
# Edit eval_qwen.sh to configure your model path and parameters
./eval_qwen.sh
```

The script will automatically run:
1. Baseline evaluation without SWIFT
2. SWIFT evaluation with optimization and Bayesian search

### Configuration

Edit `eval_qwen.sh` to customize:

```bash
MODEL_PATH=/data/models/Qwen/Qwen3-4B-Instruct-2507  # Your model path
MODEL_NAME=qwen3-4b-instruct                          # Model identifier
TASK_NAME="humaneval"                                 # Task: "cnndm" or "humaneval"
DATA_NUM=100                                          # Number of test samples
GPU_DEVICES=0                                         # GPU device ID
```

### Manual Execution (Advanced)

For fine-grained control, run components individually:

```bash
cd evaluation_qwen

# Baseline inference
python3 inference_baseline.py \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --model-id qwen3-4b-instruct \
    --task-name humaneval \
    --data-num 10

# SWIFT inference
python3 inference_swift.py \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --model-id qwen3-4b-instruct \
    --task-name humaneval \
    --data-num 10 \
    --optimization \
    --skip-ratio 0.45
```

## Configuration Options

### SWIFT Parameters
- `--skip-ratio`: Proportion of layers to skip (default: 0.45)
- `--optimization`: Enable dynamic layer optimization
- `--bayes`: Enable Bayesian optimization for skip patterns
- `--opt-interval`: Optimization interval (default: 1)
- `--context-window`: Context window for optimization (default: 32)

### Model Parameters
- `--temperature`: Sampling temperature (default: 0.0)
- `--top-p`: Top-p sampling parameter (default: 0.85)
- `--dtype`: Model precision (float16, bfloat16, float32)

## Dependencies
- **transformers>=4.51.0**
- torch
- numpy
- bayes_opt (for Bayesian optimization)
- fastchat (for utility functions)

## Future Enhancements
- Integration with Qwen3's thinking mode capabilities
- Extended evaluation on more benchmarks