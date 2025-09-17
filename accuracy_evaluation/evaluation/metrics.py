"""
Evaluation metrics for comparing baseline and SWIFT outputs.
"""

import ast
import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np
from scipy import stats
import pandas as pd

# Import evaluation libraries with fallback handling
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available")

try:
    import sacrebleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    logging.warning("sacrebleu not available")

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logging.warning("bert-score not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available")

logger = logging.getLogger(__name__)


def compute_rouge_scores(baseline_outputs: List[str], swift_outputs: List[str], 
                        references: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute ROUGE scores between baseline and SWIFT outputs.
    
    Args:
        baseline_outputs: List of baseline generated texts
        swift_outputs: List of SWIFT generated texts  
        references: Optional list of reference texts for absolute evaluation
        
    Returns:
        Dictionary containing ROUGE scores and statistics
    """
    if not ROUGE_AVAILABLE:
        return {"error": "rouge-score package not available"}
    
    if len(baseline_outputs) != len(swift_outputs):
        raise ValueError("Baseline and SWIFT outputs must have same length")
    
    results = {
        "relative_scores": {},  # SWIFT vs baseline
        "absolute_scores": {}   # Both vs references (if available)
    }
    
    # Configure ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Compute relative scores (SWIFT vs baseline)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for swift_text, baseline_text in zip(swift_outputs, baseline_outputs):
        scores = scorer.score(baseline_text, swift_text)
        for metric in rouge_scores:
            rouge_scores[metric].append(scores[metric].fmeasure)
    
    # Calculate statistics for relative scores
    for metric in rouge_scores:
        scores_array = np.array(rouge_scores[metric])
        results["relative_scores"][metric] = {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "median": float(np.median(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array))
        }
    
    # Compute absolute scores against references if available
    if references and len(references) == len(baseline_outputs):
        baseline_vs_ref = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        swift_vs_ref = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for baseline_text, swift_text, ref_text in zip(baseline_outputs, swift_outputs, references):
            baseline_scores = scorer.score(ref_text, baseline_text)
            swift_scores = scorer.score(ref_text, swift_text)
            
            for metric in baseline_vs_ref:
                baseline_vs_ref[metric].append(baseline_scores[metric].fmeasure)
                swift_vs_ref[metric].append(swift_scores[metric].fmeasure)
        
        # Calculate statistics for absolute scores
        for metric in baseline_vs_ref:
            baseline_array = np.array(baseline_vs_ref[metric])
            swift_array = np.array(swift_vs_ref[metric])
            
            results["absolute_scores"][metric] = {
                "baseline": {
                    "mean": float(np.mean(baseline_array)),
                    "std": float(np.std(baseline_array))
                },
                "swift": {
                    "mean": float(np.mean(swift_array)),
                    "std": float(np.std(swift_array))
                },
                "difference": float(np.mean(swift_array) - np.mean(baseline_array))
            }
    
    logger.info("ROUGE scores computed successfully")
    return results


def compute_bleu_scores(baseline_outputs: List[str], swift_outputs: List[str],
                       references: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute BLEU scores between baseline and SWIFT outputs.
    
    Args:
        baseline_outputs: List of baseline generated texts
        swift_outputs: List of SWIFT generated texts
        references: Optional list of reference texts
        
    Returns:
        Dictionary containing BLEU scores and statistics
    """
    if not BLEU_AVAILABLE:
        return {"error": "sacrebleu package not available"}
    
    if len(baseline_outputs) != len(swift_outputs):
        raise ValueError("Baseline and SWIFT outputs must have same length")
    
    results = {
        "relative_scores": {},
        "absolute_scores": {}
    }
    
    # Compute relative scores (SWIFT vs baseline)
    bleu_scores = []
    for swift_text, baseline_text in zip(swift_outputs, baseline_outputs):
        try:
            bleu = sacrebleu.sentence_bleu(swift_text, [baseline_text])
            bleu_scores.append(bleu.score)
        except Exception as e:
            logger.warning(f"Error computing BLEU for pair: {e}")
            bleu_scores.append(0.0)
    
    bleu_array = np.array(bleu_scores)
    results["relative_scores"] = {
        "mean": float(np.mean(bleu_array)),
        "std": float(np.std(bleu_array)),
        "median": float(np.median(bleu_array)),
        "min": float(np.min(bleu_array)),
        "max": float(np.max(bleu_array))
    }
    
    # Compute absolute scores against references if available
    if references and len(references) == len(baseline_outputs):
        baseline_bleu = []
        swift_bleu = []
        
        for baseline_text, swift_text, ref_text in zip(baseline_outputs, swift_outputs, references):
            try:
                baseline_score = sacrebleu.sentence_bleu(baseline_text, [ref_text])
                swift_score = sacrebleu.sentence_bleu(swift_text, [ref_text])
                baseline_bleu.append(baseline_score.score)
                swift_bleu.append(swift_score.score)
            except Exception as e:
                logger.warning(f"Error computing BLEU against reference: {e}")
                baseline_bleu.append(0.0)
                swift_bleu.append(0.0)
        
        baseline_array = np.array(baseline_bleu)
        swift_array = np.array(swift_bleu)
        
        results["absolute_scores"] = {
            "baseline": {
                "mean": float(np.mean(baseline_array)),
                "std": float(np.std(baseline_array))
            },
            "swift": {
                "mean": float(np.mean(swift_array)),
                "std": float(np.std(swift_array))
            },
            "difference": float(np.mean(swift_array) - np.mean(baseline_array))
        }
    
    logger.info("BLEU scores computed successfully")
    return results


def compute_bert_scores(baseline_outputs: List[str], swift_outputs: List[str],
                       references: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute BERTScore between baseline and SWIFT outputs.
    
    Args:
        baseline_outputs: List of baseline generated texts
        swift_outputs: List of SWIFT generated texts
        references: Optional list of reference texts
        
    Returns:
        Dictionary containing BERTScore results
    """
    if not BERTSCORE_AVAILABLE:
        return {"error": "bert-score package not available"}
    
    if len(baseline_outputs) != len(swift_outputs):
        raise ValueError("Baseline and SWIFT outputs must have same length")
    
    results = {
        "relative_scores": {},
        "absolute_scores": {}
    }
    
    try:
        # Compute relative scores (SWIFT vs baseline)
        P, R, F1 = bert_score(swift_outputs, baseline_outputs, lang="en", verbose=False)
        
        results["relative_scores"] = {
            "precision": {
                "mean": float(P.mean()),
                "std": float(P.std()),
                "median": float(P.median())
            },
            "recall": {
                "mean": float(R.mean()),
                "std": float(R.std()),
                "median": float(R.median())
            },
            "f1": {
                "mean": float(F1.mean()),
                "std": float(F1.std()),
                "median": float(F1.median())
            }
        }
        
        # Compute absolute scores against references if available
        if references and len(references) == len(baseline_outputs):
            P_base, R_base, F1_base = bert_score(baseline_outputs, references, lang="en", verbose=False)
            P_swift, R_swift, F1_swift = bert_score(swift_outputs, references, lang="en", verbose=False)
            
            results["absolute_scores"] = {
                "baseline": {
                    "precision": float(P_base.mean()),
                    "recall": float(R_base.mean()),
                    "f1": float(F1_base.mean())
                },
                "swift": {
                    "precision": float(P_swift.mean()),
                    "recall": float(R_swift.mean()),
                    "f1": float(F1_swift.mean())
                },
                "difference": {
                    "precision": float(P_swift.mean() - P_base.mean()),
                    "recall": float(R_swift.mean() - R_base.mean()),
                    "f1": float(F1_swift.mean() - F1_base.mean())
                }
            }
        
        logger.info("BERTScore computed successfully")
        
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        results = {"error": f"BERTScore computation failed: {e}"}
    
    return results


def compute_semantic_similarity(baseline_outputs: List[str], swift_outputs: List[str],
                               model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Compute semantic similarity between baseline and SWIFT outputs using sentence embeddings.
    
    Args:
        baseline_outputs: List of baseline generated texts
        swift_outputs: List of SWIFT generated texts
        model_name: SentenceTransformer model name
        
    Returns:
        Dictionary containing similarity scores and statistics
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return {"error": "sentence-transformers package not available"}
    
    if len(baseline_outputs) != len(swift_outputs):
        raise ValueError("Baseline and SWIFT outputs must have same length")
    
    try:
        # Load the sentence transformer model
        model = SentenceTransformer(model_name)
        
        # Encode all texts
        baseline_embeddings = model.encode(baseline_outputs)
        swift_embeddings = model.encode(swift_outputs)
        
        # Compute cosine similarities
        similarities = []
        for base_emb, swift_emb in zip(baseline_embeddings, swift_embeddings):
            similarity = np.dot(base_emb, swift_emb) / (np.linalg.norm(base_emb) * np.linalg.norm(swift_emb))
            similarities.append(float(similarity))
        
        similarities = np.array(similarities)
        
        results = {
            "cosine_similarity": {
                "mean": float(np.mean(similarities)),
                "std": float(np.std(similarities)),
                "median": float(np.median(similarities)),
                "min": float(np.min(similarities)),
                "max": float(np.max(similarities))
            },
            "model_used": model_name,
            "num_samples": len(similarities)
        }
        
        logger.info(f"Semantic similarity computed using {model_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return {"error": f"Semantic similarity computation failed: {e}"}


def compute_syntax_correctness(outputs: List[str], language: str = "python") -> Dict[str, Any]:
    """
    Compute syntax correctness rate for generated code.
    
    Args:
        outputs: List of generated code texts
        language: Programming language ("python" supported)
        
    Returns:
        Dictionary containing syntax correctness statistics
    """
    if language != "python":
        return {"error": f"Language {language} not supported yet"}
    
    syntax_correct = []
    syntax_errors = []
    
    for i, code in enumerate(outputs):
        try:
            # Try to parse the code as Python AST
            ast.parse(code)
            syntax_correct.append(True)
        except SyntaxError as e:
            syntax_correct.append(False)
            syntax_errors.append({
                "sample_index": i,
                "error": str(e),
                "line": getattr(e, 'lineno', None)
            })
        except Exception as e:
            syntax_correct.append(False)
            syntax_errors.append({
                "sample_index": i,
                "error": f"Parsing error: {str(e)}",
                "line": None
            })
    
    correctness_rate = np.mean(syntax_correct)
    
    results = {
        "syntax_correctness_rate": float(correctness_rate),
        "total_samples": len(outputs),
        "correct_samples": sum(syntax_correct),
        "error_samples": len(syntax_errors),
        "syntax_errors": syntax_errors[:10]  # Limit to first 10 errors
    }
    
    logger.info(f"Syntax correctness: {correctness_rate:.3f} ({sum(syntax_correct)}/{len(outputs)})")
    return results


def compute_pass_at_k(baseline_outputs: List[str], swift_outputs: List[str], 
                     k: int = 1, timeout: int = 10) -> Dict[str, Any]:
    """
    Compute pass@k for code generation (simplified version).
    
    Args:
        baseline_outputs: List of baseline generated code
        swift_outputs: List of SWIFT generated code
        k: Number of attempts (simplified to 1 for now)
        timeout: Timeout for code execution in seconds
        
    Returns:
        Dictionary containing pass@k results
        
    Note:
        This is a simplified implementation. Full pass@k requires test cases
        and proper execution environment setup.
    """
    if len(baseline_outputs) != len(swift_outputs):
        raise ValueError("Baseline and SWIFT outputs must have same length")
    
    baseline_syntax = compute_syntax_correctness(baseline_outputs)
    swift_syntax = compute_syntax_correctness(swift_outputs)
    
    results = {
        "note": "Simplified pass@k implementation using syntax correctness",
        "baseline_syntax_rate": baseline_syntax["syntax_correctness_rate"],
        "swift_syntax_rate": swift_syntax["syntax_correctness_rate"],
        "syntax_rate_difference": (
            swift_syntax["syntax_correctness_rate"] - 
            baseline_syntax["syntax_correctness_rate"]
        ),
        "total_samples": len(baseline_outputs)
    }
    
    logger.info("Pass@k computed (simplified version using syntax correctness)")
    return results


def test_distribution_preservation(baseline_outputs: List[str], swift_outputs: List[str]) -> Dict[str, Any]:
    """
    Test whether SWIFT preserves the output distribution compared to baseline.
    
    Args:
        baseline_outputs: List of baseline generated texts
        swift_outputs: List of SWIFT generated texts
        
    Returns:
        Dictionary containing distribution preservation test results
    """
    if len(baseline_outputs) != len(swift_outputs):
        raise ValueError("Baseline and SWIFT outputs must have same length")
    
    results = {}
    
    # Length distribution comparison
    baseline_lengths = [len(text.split()) for text in baseline_outputs]
    swift_lengths = [len(text.split()) for text in swift_outputs]
    
    # Statistical tests for length distribution
    length_ks_stat, length_p_value = stats.ks_2samp(baseline_lengths, swift_lengths)
    length_mannwhitney_stat, length_mannwhitney_p = stats.mannwhitneyu(
        baseline_lengths, swift_lengths, alternative='two-sided'
    )
    
    results["length_distribution"] = {
        "baseline_mean": float(np.mean(baseline_lengths)),
        "swift_mean": float(np.mean(swift_lengths)),
        "baseline_std": float(np.std(baseline_lengths)),
        "swift_std": float(np.std(swift_lengths)),
        "ks_test": {
            "statistic": float(length_ks_stat),
            "p_value": float(length_p_value),
            "significant": length_p_value < 0.05
        },
        "mann_whitney_test": {
            "statistic": float(length_mannwhitney_stat),
            "p_value": float(length_mannwhitney_p),
            "significant": length_mannwhitney_p < 0.05
        }
    }
    
    # Vocabulary diversity comparison
    baseline_vocab = set()
    swift_vocab = set()
    
    for text in baseline_outputs:
        baseline_vocab.update(text.lower().split())
    
    for text in swift_outputs:
        swift_vocab.update(text.lower().split())
    
    vocab_overlap = len(baseline_vocab.intersection(swift_vocab))
    vocab_union = len(baseline_vocab.union(swift_vocab))
    
    results["vocabulary_diversity"] = {
        "baseline_vocab_size": len(baseline_vocab),
        "swift_vocab_size": len(swift_vocab),
        "vocab_overlap": vocab_overlap,
        "vocab_union": vocab_union,
        "jaccard_similarity": float(vocab_overlap / vocab_union) if vocab_union > 0 else 0.0
    }
    
    # Repetition analysis
    baseline_repetition = _compute_repetition_rate(baseline_outputs)
    swift_repetition = _compute_repetition_rate(swift_outputs)
    
    results["repetition_analysis"] = {
        "baseline_repetition_rate": baseline_repetition,
        "swift_repetition_rate": swift_repetition,
        "repetition_difference": swift_repetition - baseline_repetition
    }
    
    logger.info("Distribution preservation tests completed")
    return results


def _compute_repetition_rate(outputs: List[str]) -> float:
    """
    Compute the repetition rate in generated outputs.
    
    Args:
        outputs: List of generated texts
        
    Returns:
        Average repetition rate across all outputs
    """
    repetition_rates = []
    
    for text in outputs:
        tokens = text.split()
        if len(tokens) < 2:
            repetition_rates.append(0.0)
            continue
        
        # Count repeated tokens
        token_counts = Counter(tokens)
        repeated_tokens = sum(count - 1 for count in token_counts.values() if count > 1)
        repetition_rate = repeated_tokens / len(tokens) if len(tokens) > 0 else 0.0
        repetition_rates.append(repetition_rate)
    
    return float(np.mean(repetition_rates))


def compute_comprehensive_metrics(baseline_outputs: List[str], swift_outputs: List[str],
                                task_name: str, references: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute all available metrics for the given task.
    
    Args:
        baseline_outputs: List of baseline generated texts
        swift_outputs: List of SWIFT generated texts
        task_name: Name of the task ("cnndm", "humaneval", etc.)
        references: Optional list of reference texts
        
    Returns:
        Dictionary containing all computed metrics
    """
    results = {
        "task_name": task_name,
        "num_samples": len(baseline_outputs),
        "metrics": {}
    }
    
    # Common metrics for all tasks
    results["metrics"]["semantic_similarity"] = compute_semantic_similarity(baseline_outputs, swift_outputs)
    results["metrics"]["distribution_preservation"] = test_distribution_preservation(baseline_outputs, swift_outputs)
    
    # Task-specific metrics
    if task_name in ["cnndm", "summarization"]:
        results["metrics"]["rouge"] = compute_rouge_scores(baseline_outputs, swift_outputs, references)
        results["metrics"]["bleu"] = compute_bleu_scores(baseline_outputs, swift_outputs, references)
        results["metrics"]["bert_score"] = compute_bert_scores(baseline_outputs, swift_outputs, references)
        
    elif task_name in ["humaneval", "code_generation"]:
        results["metrics"]["syntax_correctness"] = {
            "baseline": compute_syntax_correctness(baseline_outputs),
            "swift": compute_syntax_correctness(swift_outputs)
        }
        results["metrics"]["pass_at_k"] = compute_pass_at_k(baseline_outputs, swift_outputs)
    
    logger.info(f"Comprehensive metrics computed for {task_name}")
    return results