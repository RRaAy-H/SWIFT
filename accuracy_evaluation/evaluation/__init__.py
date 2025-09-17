"""
SWIFT Accuracy Evaluation Package

This package provides comprehensive evaluation metrics for comparing 
baseline and SWIFT LLM inference outputs across different tasks.
"""

from .utils import load_jsonl_data, validate_data_format, extract_outputs, filter_valid_samples, align_outputs, get_task_name_from_path
from .metrics import (
    compute_rouge_scores,
    compute_bleu_scores, 
    compute_bert_scores,
    compute_pass_at_k,
    compute_syntax_correctness,
    compute_semantic_similarity,
    compute_comprehensive_metrics,
    test_distribution_preservation
)
from .report import generate_evaluation_report,  create_comparison_table, save_results

__all__ = [
    'load_jsonl_data',
    'validate_data_format', 
    'extract_outputs',
    'filter_valid_samples',
    'align_outputs',
    'get_task_name_from_path',
    'compute_rouge_scores',
    'compute_bleu_scores',
    'compute_bert_scores', 
    'compute_pass_at_k',
    'compute_syntax_correctness',
    'compute_semantic_similarity',
    'compute_comprehensive_metrics',
    'test_distribution_preservation',
    'generate_evaluation_report',
    'create_comparison_table',
    'save_results'
]
