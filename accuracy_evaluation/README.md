# SWIFT Accuracy Evaluation Pipeline

A comprehensive evaluation framework for comparing accuracy and output quality between baseline and SWIFT LLM inference implementations.

## Overview

This pipeline provides automated evaluation of SWIFT's performance across multiple dimensions:

- **Semantic Similarity**: How similar are the outputs semantically?
- **Task-Specific Metrics**: ROUGE/BLEU for summarization, pass@k for code generation
- **Distribution Preservation**: Does SWIFT maintain the same output characteristics?
- **Quality Assessment**: Comprehensive analysis across multiple quality dimensions

## Features

✅ **Multi-Task Support**: CNN/DailyMail (summarization), HumanEval (code generation)  
✅ **Comprehensive Metrics**: ROUGE, BLEU, BERTScore, semantic similarity, syntax checking  
✅ **Statistical Analysis**: Distribution preservation tests, significance testing  
✅ **Automated Reports**: Detailed evaluation reports with conclusions  
✅ **Robust Error Handling**: Graceful handling of malformed data and missing dependencies  
✅ **CLI Interface**: Easy-to-use command-line interface with extensive options  

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make the script executable (optional)
chmod +x evaluate_accuracy.py
```

## Quick Start

### Basic Usage

```bash
# Evaluate summarization task
python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task cnndm

# Evaluate code generation task  
python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task humaneval

# Auto-detect task from filename
python evaluate_accuracy.py --baseline cnndm_baseline.jsonl --swift cnndm_swift.jsonl
```

### With Output Directory

```bash
python evaluate_accuracy.py \
    --baseline outputs/cnndm_baseline.jsonl \
    --swift outputs/cnndm_swift.jsonl \
    --task cnndm \
    --output-dir evaluation_results/ \
    --print-summary
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--baseline` | Path to baseline output JSONL file | **Required** |
| `--swift` | Path to SWIFT output JSONL file | **Required** |
| `--task` | Task type: `cnndm`, `humaneval`, `auto` | `auto` |
| `--references` | Path to reference data (optional) | None |
| `--output-dir` | Directory to save results | `results` |
| `--max-samples` | Maximum samples to evaluate | All |
| `--log-level` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--no-report` | Skip generating text report | False |
| `--no-save-json` | Skip saving JSON results | False |
| `--print-summary` | Print summary table to console | False |

## Expected Input Format

Your JSONL files should follow the SWIFT evaluation format:

```json
{
  "model_id": "llama-2-13b",
  "choices": [
    {
      "turns": "Generated text output here...",
      "new_tokens": [45],
      "wall_time": [2.3],
      "accept_lengths": [1, 1, 1, ...]
    }
  ],
  "tstamp": 1699123456.789
}
```

## Output

The pipeline generates several outputs:

### 1. Console Summary
```
KEY RESULTS:
----------------------------------------
Semantic Similarity: 0.8542
ROUGE-1 Similarity: 0.7831
BERTScore F1: 0.9012
Length Distribution Differs: False
```

### 2. Detailed Report (`swift_evaluation_report_YYYYMMDD_HHMMSS.txt`)
- Executive summary with key findings
- Detailed metric breakdowns
- Statistical analysis results  
- Conclusions and recommendations

### 3. JSON Results (`evaluation_results.json`)
- Complete evaluation data in structured format
- All computed metrics and statistics
- Metadata about the evaluation run

## Supported Metrics

### Summarization Tasks (CNN/DailyMail)

| Metric | Description | Purpose |
|--------|-------------|---------|
| **ROUGE-1/2/L** | Lexical overlap with baseline | Content preservation |
| **BLEU** | N-gram precision | Translation-style quality |
| **BERTScore** | Semantic similarity using BERT | Meaning preservation |
| **Semantic Similarity** | Sentence embedding cosine similarity | Overall semantic alignment |

### Code Generation Tasks (HumanEval)

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Syntax Correctness** | Valid Python syntax rate | Basic code quality |
| **Pass@k** | Functional correctness (simplified) | Code functionality |
| **Semantic Similarity** | Code semantic similarity | Logic preservation |

### Universal Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **Distribution Preservation** | Statistical tests for output distribution | Verify SWIFT losslessness |
| **Length Distribution** | Output length comparison | Structural similarity |
| **Vocabulary Diversity** | Unique word usage analysis | Content richness |
| **Repetition Analysis** | Token repetition patterns | Output quality |

## Examples

### Example 1: Basic CNN/DailyMail Evaluation

```bash
python evaluate_accuracy.py \
    --baseline outputs/cnndm_100/model_answer/llama-2-13b/llama-2-13b-vanilla.jsonl \
    --swift outputs/cnndm_100/model_answer/llama-2-13b/llama-2-13b-swift.jsonl \
    --task cnndm \
    --output-dir cnndm_evaluation/
```

### Example 2: HumanEval with Limited Samples

```bash
python evaluate_accuracy.py \
    --baseline outputs/humaneval_100/model_answer/codellama-13b/codellama-13b-vanilla.jsonl \
    --swift outputs/humaneval_100/model_answer/codellama-13b/codellama-13b-swift.jsonl \
    --task humaneval \
    --max-samples 50 \
    --print-summary \
    --log-level DEBUG
```

### Example 3: With Reference Data

```bash
python evaluate_accuracy.py \
    --baseline baseline.jsonl \
    --swift swift.jsonl \
    --references references.jsonl \
    --task cnndm \
    --output-dir detailed_evaluation/
```

## Understanding Results

### Semantic Similarity Scores
- **0.9+ (Excellent)**: SWIFT preserves meaning very well
- **0.8-0.9 (Good)**: Acceptable semantic preservation  
- **0.7-0.8 (Moderate)**: Some semantic differences
- **<0.7 (Poor)**: Significant semantic changes

### ROUGE Scores (Summarization)
- **0.8+ (Excellent)**: High content overlap
- **0.6-0.8 (Good)**: Reasonable content preservation
- **0.4-0.6 (Moderate)**: Some content differences
- **<0.4 (Poor)**: Significant content changes

### Syntax Correctness (Code)
- **0.95+ (Excellent)**: Very reliable code generation
- **0.9-0.95 (Good)**: Mostly correct syntax
- **0.8-0.9 (Moderate)**: Some syntax issues
- **<0.8 (Poor)**: Many syntax errors

## Troubleshooting

### Common Issues

1. **ImportError: Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **FileNotFoundError: JSONL file not found**
   - Check file paths are correct
   - Ensure files exist and are readable

3. **ValueError: Invalid JSON format**
   - Verify JSONL files are properly formatted
   - Check for corrupted lines in files

4. **Memory issues with large files**
   - Use `--max-samples` to limit evaluation size
   - Process files in smaller batches

### Debug Mode

For detailed troubleshooting, use debug mode:

```bash
python evaluate_accuracy.py \
    --baseline baseline.jsonl \
    --swift swift.jsonl \
    --log-level DEBUG
```

## Architecture

The pipeline consists of several modular components:

```
accuracy_evaluation/
├── requirements.txt          # Dependencies
├── evaluate_accuracy.py      # Main CLI script  
├── evaluation/              # Core evaluation package
│   ├── __init__.py          # Package initialization
│   ├── utils.py             # Data loading and validation
│   ├── metrics.py           # Evaluation metrics computation
│   └── report.py            # Report generation
└── README.md                # This documentation
```