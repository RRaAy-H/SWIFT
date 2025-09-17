"""
SWIFT Accuracy Evaluation Package

This package provides comprehensive evaluation metrics for comparing 
baseline and SWIFT LLM inference outputs across different tasks.
"""

__version__ = "1.0.0"
__author__ = "SWIFT Evaluation Team"

from .utils import load_jsonl_data, validate_data_format, extract_outputs
from .metrics import (
    compute_rouge_scores,
    compute_bleu_scores, 
    compute_bert_scores,
    compute_pass_at_k,
    compute_syntax_correctness,
    compute_semantic_similarity,
    test_distribution_preservation
)
from .report import generate_evaluation_report, save_results

__all__ = [
    'load_jsonl_data',
    'validate_data_format', 
    'extract_outputs',
    'compute_rouge_scores',
    'compute_bleu_scores',
    'compute_bert_scores', 
    'compute_pass_at_k',
    'compute_syntax_correctness',
    'compute_semantic_similarity',
    'test_distribution_preservation',
    'generate_evaluation_report',
    'save_results'
]