"""
Report generation functions for SWIFT accuracy evaluation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    logging.warning("tabulate not available - tables will be basic")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("matplotlib/seaborn not available - no plots will be generated")

logger = logging.getLogger(__name__)


def generate_evaluation_report(results: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """
    Generate a comprehensive evaluation report from results.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save the report (optional)
        
    Returns:
        String containing the formatted report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("SWIFT ACCURACY EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Task: {results.get('task_name', 'Unknown')}")
    report_lines.append(f"Samples: {results.get('num_samples', 'Unknown')}")
    report_lines.append("")
    
    # Summary
    summary = _generate_summary(results)
    report_lines.extend(summary)
    report_lines.append("")
    
    # Detailed metrics
    if "metrics" in results:
        for metric_name, metric_data in results["metrics"].items():
            if isinstance(metric_data, dict) and "error" not in metric_data:
                section = _generate_metric_section(metric_name, metric_data)
                report_lines.extend(section)
                report_lines.append("")
    
    # Conclusions and recommendations
    conclusions = _generate_conclusions(results)
    report_lines.extend(conclusions)
    
    report_text = "\n".join(report_lines)
    
    # Save to file if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"swift_evaluation_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_file}")
    
    return report_text


def _generate_summary(results: Dict[str, Any]) -> list:
    """Generate executive summary section."""
    lines = []
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)
    
    task_name = results.get('task_name', 'unknown')
    metrics = results.get('metrics', {})
    
    # Extract key findings
    key_findings = []
    
    # Semantic similarity
    if 'semantic_similarity' in metrics:
        sim_data = metrics['semantic_similarity']
        if 'cosine_similarity' in sim_data:
            sim_mean = sim_data['cosine_similarity']['mean']
            key_findings.append(f"• Semantic similarity: {sim_mean:.3f} (cosine similarity)")
    
    # Task-specific findings
    if task_name in ['cnndm', 'summarization']:
        if 'rouge' in metrics and 'relative_scores' in metrics['rouge']:
            rouge_data = metrics['rouge']['relative_scores']
            if 'rouge1' in rouge_data:
                rouge1_mean = rouge_data['rouge1']['mean']
                key_findings.append(f"• ROUGE-1 similarity: {rouge1_mean:.3f}")
        
        if 'bert_score' in metrics and 'relative_scores' in metrics['bert_score']:
            bert_data = metrics['bert_score']['relative_scores']
            if 'f1' in bert_data:
                bert_f1 = bert_data['f1']['mean']
                key_findings.append(f"• BERTScore F1: {bert_f1:.3f}")
    
    elif task_name in ['humaneval', 'code_generation']:
        if 'syntax_correctness' in metrics:
            baseline_syntax = metrics['syntax_correctness'].get('baseline', {})
            swift_syntax = metrics['syntax_correctness'].get('swift', {})
            
            baseline_rate = baseline_syntax.get('syntax_correctness_rate', 0)
            swift_rate = swift_syntax.get('syntax_correctness_rate', 0)
            
            key_findings.append(f"• Baseline syntax correctness: {baseline_rate:.3f}")
            key_findings.append(f"• SWIFT syntax correctness: {swift_rate:.3f}")
    
    # Distribution preservation
    if 'distribution_preservation' in metrics:
        dist_data = metrics['distribution_preservation']
        if 'length_distribution' in dist_data:
            length_data = dist_data['length_distribution']
            ks_significant = length_data.get('ks_test', {}).get('significant', False)
            key_findings.append(f"• Length distribution differs: {ks_significant}")
    
    if key_findings:
        lines.extend(key_findings)
    else:
        lines.append("• No key findings available")
    
    lines.append("")
    
    # Overall assessment
    assessment = _assess_overall_quality(results)
    lines.append(f"Overall Assessment: {assessment}")
    
    return lines


def _generate_metric_section(metric_name: str, metric_data: Dict[str, Any]) -> list:
    """Generate detailed section for a specific metric."""
    lines = []
    lines.append(f"{metric_name.upper().replace('_', ' ')}")
    lines.append("-" * len(metric_name))
    
    if metric_name == "semantic_similarity":
        lines.extend(_format_semantic_similarity(metric_data))
    elif metric_name == "rouge":
        lines.extend(_format_rouge_scores(metric_data))
    elif metric_name == "bleu":
        lines.extend(_format_bleu_scores(metric_data))
    elif metric_name == "bert_score":
        lines.extend(_format_bert_scores(metric_data))
    elif metric_name == "syntax_correctness":
        lines.extend(_format_syntax_correctness(metric_data))
    elif metric_name == "distribution_preservation":
        lines.extend(_format_distribution_preservation(metric_data))
    else:
        # Generic formatting
        lines.append(json.dumps(metric_data, indent=2))
    
    return lines


def _format_semantic_similarity(data: Dict[str, Any]) -> list:
    """Format semantic similarity results."""
    lines = []
    
    if 'cosine_similarity' in data:
        sim_data = data['cosine_similarity']
        lines.append(f"Cosine Similarity Statistics:")
        lines.append(f"  Mean:   {sim_data['mean']:.4f}")
        lines.append(f"  Std:    {sim_data['std']:.4f}")
        lines.append(f"  Median: {sim_data['median']:.4f}")
        lines.append(f"  Range:  [{sim_data['min']:.4f}, {sim_data['max']:.4f}]")
        
        # Interpretation
        mean_sim = sim_data['mean']
        if mean_sim > 0.9:
            interpretation = "Very high similarity - SWIFT preserves content well"
        elif mean_sim > 0.8:
            interpretation = "High similarity - Good content preservation"
        elif mean_sim > 0.7:
            interpretation = "Moderate similarity - Some content changes"
        else:
            interpretation = "Low similarity - Significant content differences"
        
        lines.append(f"  Interpretation: {interpretation}")
    
    if 'model_used' in data:
        lines.append(f"Model: {data['model_used']}")
    
    return lines


def _format_rouge_scores(data: Dict[str, Any]) -> list:
    """Format ROUGE score results."""
    lines = []
    
    if 'relative_scores' in data:
        lines.append("SWIFT vs Baseline Comparison:")
        rel_data = data['relative_scores']
        
        if TABULATE_AVAILABLE:
            table_data = []
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                if metric in rel_data:
                    scores = rel_data[metric]
                    table_data.append([
                        metric.upper(),
                        f"{scores['mean']:.4f}",
                        f"{scores['std']:.4f}",
                        f"[{scores['min']:.4f}, {scores['max']:.4f}]"
                    ])
            
            if table_data:
                headers = ["Metric", "Mean", "Std", "Range"]
                lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            for metric in ['rouge1', 'rouge2', 'rougeL']:
                if metric in rel_data:
                    scores = rel_data[metric]
                    lines.append(f"  {metric.upper()}: {scores['mean']:.4f} ± {scores['std']:.4f}")
    
    if 'absolute_scores' in data:
        lines.append("\nAgainst References:")
        abs_data = data['absolute_scores']
        
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            if metric in abs_data:
                metric_data = abs_data[metric]
                baseline_score = metric_data.get('baseline', {}).get('mean', 0)
                swift_score = metric_data.get('swift', {}).get('mean', 0)
                diff = metric_data.get('difference', 0)
                
                lines.append(f"  {metric.upper()}:")
                lines.append(f"    Baseline: {baseline_score:.4f}")
                lines.append(f"    SWIFT:    {swift_score:.4f}")
                lines.append(f"    Diff:     {diff:+.4f}")
    
    return lines


def _format_bleu_scores(data: Dict[str, Any]) -> list:
    """Format BLEU score results."""
    lines = []
    
    if 'relative_scores' in data:
        rel_data = data['relative_scores']
        lines.append("SWIFT vs Baseline BLEU:")
        lines.append(f"  Mean:   {rel_data['mean']:.4f}")
        lines.append(f"  Std:    {rel_data['std']:.4f}")
        lines.append(f"  Median: {rel_data['median']:.4f}")
        lines.append(f"  Range:  [{rel_data['min']:.4f}, {rel_data['max']:.4f}]")
    
    if 'absolute_scores' in data:
        abs_data = data['absolute_scores']
        lines.append("\nAgainst References:")
        lines.append(f"  Baseline: {abs_data['baseline']['mean']:.4f}")
        lines.append(f"  SWIFT:    {abs_data['swift']['mean']:.4f}")
        lines.append(f"  Diff:     {abs_data['difference']:+.4f}")
    
    return lines


def _format_bert_scores(data: Dict[str, Any]) -> list:
    """Format BERTScore results."""
    lines = []
    
    if 'relative_scores' in data:
        rel_data = data['relative_scores']
        lines.append("SWIFT vs Baseline BERTScore:")
        
        for component in ['precision', 'recall', 'f1']:
            if component in rel_data:
                comp_data = rel_data[component]
                lines.append(f"  {component.capitalize()}:")
                lines.append(f"    Mean:   {comp_data['mean']:.4f}")
                lines.append(f"    Std:    {comp_data['std']:.4f}")
                lines.append(f"    Median: {comp_data['median']:.4f}")
    
    if 'absolute_scores' in data:
        abs_data = data['absolute_scores']
        lines.append("\nAgainst References:")
        
        for component in ['precision', 'recall', 'f1']:
            if component in abs_data['baseline']:
                baseline_score = abs_data['baseline'][component]
                swift_score = abs_data['swift'][component]
                diff = abs_data['difference'][component]
                
                lines.append(f"  {component.capitalize()}:")
                lines.append(f"    Baseline: {baseline_score:.4f}")
                lines.append(f"    SWIFT:    {swift_score:.4f}")
                lines.append(f"    Diff:     {diff:+.4f}")
    
    return lines


def _format_syntax_correctness(data: Dict[str, Any]) -> list:
    """Format syntax correctness results."""
    lines = []
    
    if 'baseline' in data and 'swift' in data:
        baseline_data = data['baseline']
        swift_data = data['swift']
        
        baseline_rate = baseline_data.get('syntax_correctness_rate', 0)
        swift_rate = swift_data.get('syntax_correctness_rate', 0)
        
        lines.append("Syntax Correctness Comparison:")
        lines.append(f"  Baseline: {baseline_rate:.4f} ({baseline_data.get('correct_samples', 0)}/{baseline_data.get('total_samples', 0)})")
        lines.append(f"  SWIFT:    {swift_rate:.4f} ({swift_data.get('correct_samples', 0)}/{swift_data.get('total_samples', 0)})")
        lines.append(f"  Difference: {swift_rate - baseline_rate:+.4f}")
        
        # Show some syntax errors if available
        if 'syntax_errors' in baseline_data and baseline_data['syntax_errors']:
            lines.append("\nSample Baseline Syntax Errors:")
            for error in baseline_data['syntax_errors'][:3]:
                lines.append(f"  Sample {error['sample_index']}: {error['error']}")
        
        if 'syntax_errors' in swift_data and swift_data['syntax_errors']:
            lines.append("\nSample SWIFT Syntax Errors:")
            for error in swift_data['syntax_errors'][:3]:
                lines.append(f"  Sample {error['sample_index']}: {error['error']}")
    
    return lines


def _format_distribution_preservation(data: Dict[str, Any]) -> list:
    """Format distribution preservation test results."""
    lines = []
    
    # Length distribution
    if 'length_distribution' in data:
        length_data = data['length_distribution']
        lines.append("Length Distribution Analysis:")
        lines.append(f"  Baseline: {length_data['baseline_mean']:.2f} ± {length_data['baseline_std']:.2f} words")
        lines.append(f"  SWIFT:    {length_data['swift_mean']:.2f} ± {length_data['swift_std']:.2f} words")
        
        if 'ks_test' in length_data:
            ks_data = length_data['ks_test']
            lines.append(f"  KS Test: p-value = {ks_data['p_value']:.4f} {'(significant)' if ks_data['significant'] else '(not significant)'}")
    
    # Vocabulary diversity
    if 'vocabulary_diversity' in data:
        vocab_data = data['vocabulary_diversity']
        lines.append("\nVocabulary Diversity:")
        lines.append(f"  Baseline vocab: {vocab_data['baseline_vocab_size']} unique words")
        lines.append(f"  SWIFT vocab:    {vocab_data['swift_vocab_size']} unique words")
        lines.append(f"  Jaccard sim:    {vocab_data['jaccard_similarity']:.4f}")
    
    # Repetition analysis
    if 'repetition_analysis' in data:
        rep_data = data['repetition_analysis']
        lines.append("\nRepetition Analysis:")
        lines.append(f"  Baseline: {rep_data['baseline_repetition_rate']:.4f}")
        lines.append(f"  SWIFT:    {rep_data['swift_repetition_rate']:.4f}")
        lines.append(f"  Difference: {rep_data['repetition_difference']:+.4f}")
    
    return lines


def _assess_overall_quality(results: Dict[str, Any]) -> str:
    """Generate overall quality assessment."""
    metrics = results.get('metrics', {})
    
    # Collect key quality indicators
    indicators = []
    
    # Semantic similarity
    if 'semantic_similarity' in metrics:
        sim_data = metrics['semantic_similarity']
        if 'cosine_similarity' in sim_data:
            sim_score = sim_data['cosine_similarity']['mean']
            indicators.append(('semantic_similarity', sim_score))
    
    # ROUGE scores (for summarization)
    if 'rouge' in metrics and 'relative_scores' in metrics['rouge']:
        rouge_data = metrics['rouge']['relative_scores']
        if 'rouge1' in rouge_data:
            rouge_score = rouge_data['rouge1']['mean']
            indicators.append(('rouge1', rouge_score))
    
    # Syntax correctness (for code)
    if 'syntax_correctness' in metrics:
        syntax_data = metrics['syntax_correctness']
        if 'baseline' in syntax_data and 'swift' in syntax_data:
            baseline_rate = syntax_data['baseline'].get('syntax_correctness_rate', 0)
            swift_rate = syntax_data['swift'].get('syntax_correctness_rate', 0)
            indicators.append(('syntax_preservation', abs(swift_rate - baseline_rate) < 0.05))
    
    # Make assessment based on indicators
    if not indicators:
        return "Insufficient data for assessment"
    
    high_quality_count = 0
    total_indicators = len(indicators)
    
    for name, value in indicators:
        if name == 'syntax_preservation' and isinstance(value, bool):
            if value:
                high_quality_count += 1
        elif isinstance(value, (int, float)):
            if value > 0.8:  # High similarity threshold
                high_quality_count += 1
            elif value > 0.7:  # Moderate similarity
                high_quality_count += 0.5
    
    quality_ratio = high_quality_count / total_indicators
    
    if quality_ratio >= 0.8:
        return "EXCELLENT - SWIFT preserves output quality very well"
    elif quality_ratio >= 0.6:
        return "GOOD - SWIFT maintains acceptable output quality"
    elif quality_ratio >= 0.4:
        return "MODERATE - Some quality degradation observed"
    else:
        return "POOR - Significant quality differences detected"


def _generate_conclusions(results: Dict[str, Any]) -> list:
    """Generate conclusions and recommendations."""
    lines = []
    lines.append("CONCLUSIONS AND RECOMMENDATIONS")
    lines.append("-" * 40)
    
    task_name = results.get('task_name', 'unknown')
    metrics = results.get('metrics', {})
    
    # Task-specific conclusions
    if task_name in ['cnndm', 'summarization']:
        lines.extend(_summarization_conclusions(metrics))
    elif task_name in ['humaneval', 'code_generation']:
        lines.extend(_code_generation_conclusions(metrics))
    
    # General conclusions
    lines.append("\nGeneral Recommendations:")
    
    # Semantic similarity assessment
    if 'semantic_similarity' in metrics:
        sim_data = metrics['semantic_similarity']
        if 'cosine_similarity' in sim_data:
            sim_mean = sim_data['cosine_similarity']['mean']
            if sim_mean > 0.9:
                lines.append("• SWIFT shows excellent semantic preservation")
            elif sim_mean > 0.8:
                lines.append("• SWIFT shows good semantic preservation")
            else:
                lines.append("• Consider investigating semantic preservation issues")
    
    # Distribution preservation
    if 'distribution_preservation' in metrics:
        dist_data = metrics['distribution_preservation']
        if 'length_distribution' in dist_data:
            length_data = dist_data['length_distribution']
            if length_data.get('ks_test', {}).get('significant', False):
                lines.append("• Length distributions differ significantly - investigate cause")
            else:
                lines.append("• Length distributions are well preserved")
    
    lines.append("• Continue monitoring accuracy metrics alongside speed improvements")
    lines.append("• Consider A/B testing with human evaluators for critical applications")
    
    return lines


def _summarization_conclusions(metrics: Dict[str, Any]) -> list:
    """Generate conclusions specific to summarization tasks."""
    lines = []
    lines.append("Summarization-Specific Findings:")
    
    # ROUGE assessment
    if 'rouge' in metrics and 'relative_scores' in metrics['rouge']:
        rouge_data = metrics['rouge']['relative_scores']
        rouge1_mean = rouge_data.get('rouge1', {}).get('mean', 0)
        
        if rouge1_mean > 0.8:
            lines.append("• ROUGE scores indicate excellent content overlap")
        elif rouge1_mean > 0.6:
            lines.append("• ROUGE scores indicate good content overlap")
        else:
            lines.append("• ROUGE scores suggest significant content differences")
    
    # BERTScore assessment
    if 'bert_score' in metrics and 'relative_scores' in metrics['bert_score']:
        bert_data = metrics['bert_score']['relative_scores']
        bert_f1 = bert_data.get('f1', {}).get('mean', 0)
        
        if bert_f1 > 0.9:
            lines.append("• BERTScore indicates excellent semantic similarity")
        elif bert_f1 > 0.8:
            lines.append("• BERTScore indicates good semantic similarity")
        else:
            lines.append("• BERTScore suggests semantic differences need investigation")
    
    return lines


def _code_generation_conclusions(metrics: Dict[str, Any]) -> list:
    """Generate conclusions specific to code generation tasks."""
    lines = []
    lines.append("Code Generation-Specific Findings:")
    
    if 'syntax_correctness' in metrics:
        syntax_data = metrics['syntax_correctness']
        if 'baseline' in syntax_data and 'swift' in syntax_data:
            baseline_rate = syntax_data['baseline'].get('syntax_correctness_rate', 0)
            swift_rate = syntax_data['swift'].get('syntax_correctness_rate', 0)
            diff = swift_rate - baseline_rate
            
            if abs(diff) < 0.02:
                lines.append("• Syntax correctness is well preserved")
            elif diff > 0:
                lines.append(f"• SWIFT shows improved syntax correctness (+{diff:.3f})")
            else:
                lines.append(f"• SWIFT shows reduced syntax correctness ({diff:.3f})")
    
    return lines


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results_with_metadata = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "evaluation_version": "1.0.0"
        },
        **results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


def create_comparison_table(results: Dict[str, Any]) -> str:
    """
    Create a summary comparison table.
    
    Args:
        results: Dictionary containing evaluation results
        
    Returns:
        Formatted table string
    """
    if not TABULATE_AVAILABLE:
        return "Tabulate library not available for table formatting"
    
    table_data = []
    metrics = results.get('metrics', {})
    
    # Semantic similarity
    if 'semantic_similarity' in metrics:
        sim_data = metrics['semantic_similarity']
        if 'cosine_similarity' in sim_data:
            sim_mean = sim_data['cosine_similarity']['mean']
            table_data.append(['Semantic Similarity', f"{sim_mean:.4f}", 'Cosine similarity'])
    
    # ROUGE-1 (if available)
    if 'rouge' in metrics and 'relative_scores' in metrics['rouge']:
        rouge_data = metrics['rouge']['relative_scores']
        if 'rouge1' in rouge_data:
            rouge1_mean = rouge_data['rouge1']['mean']
            table_data.append(['ROUGE-1', f"{rouge1_mean:.4f}", 'Lexical overlap'])
    
    # BERTScore F1 (if available)
    if 'bert_score' in metrics and 'relative_scores' in metrics['bert_score']:
        bert_data = metrics['bert_score']['relative_scores']
        if 'f1' in bert_data:
            bert_f1 = bert_data['f1']['mean']
            table_data.append(['BERTScore F1', f"{bert_f1:.4f}", 'Semantic similarity'])
    
    if table_data:
        headers = ['Metric', 'Score', 'Description']
        return tabulate(table_data, headers=headers, tablefmt="grid")
    else:
        return "No data available for comparison table"