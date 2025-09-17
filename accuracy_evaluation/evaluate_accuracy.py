#!/usr/bin/env python3
"""
SWIFT Accuracy Evaluation Pipeline

This script provides comprehensive evaluation of accuracy and output quality
between baseline and SWIFT LLM inference implementations.

Usage:
    python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task cnndm
    python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task humaneval --output-dir results/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import json

# Add the evaluation package to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from evaluation import (
        load_jsonl_data,
        validate_data_format,
        extract_outputs,
        filter_valid_samples,
        align_outputs,
        get_task_name_from_path,
        compute_comprehensive_metrics,
        generate_evaluation_report,
        save_results,
        create_comparison_table
    )
except ImportError as e:
    print(f"Error importing evaluation modules: {e}")
    print("Please ensure all requirements are installed: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy between baseline and SWIFT LLM outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation for summarization task
  python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task cnndm

  # Code generation evaluation with output directory
  python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task humaneval --output-dir results/

  # Auto-detect task from filename
  python evaluate_accuracy.py --baseline cnndm_baseline.jsonl --swift cnndm_swift.jsonl

  # Limit number of samples and set custom log level
  python evaluate_accuracy.py --baseline baseline.jsonl --swift swift.jsonl --task cnndm --max-samples 100 --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline output JSONL file"
    )
    
    parser.add_argument(
        "--swift", 
        type=str,
        required=True,
        help="Path to SWIFT output JSONL file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["cnndm", "humaneval", "summarization", "code_generation", "auto"],
        default="auto",
        help="Task type (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--references",
        type=str,
        help="Path to reference data JSONL file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results (default: results)"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to evaluate (default: all)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip generating the text report"
    )
    
    parser.add_argument(
        "--no-save-json",
        action="store_true", 
        help="Skip saving results to JSON"
    )
    
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print summary table to console"
    )
    
    return parser.parse_args()


def validate_files(baseline_path: str, swift_path: str, references_path: Optional[str] = None) -> None:
    """
    Validate that input files exist and are accessible.
    
    Args:
        baseline_path: Path to baseline file
        swift_path: Path to SWIFT file
        references_path: Optional path to references file
        
    Raises:
        FileNotFoundError: If any required file doesn't exist
    """
    baseline_file = Path(baseline_path)
    swift_file = Path(swift_path)
    
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
    
    if not swift_file.exists():
        raise FileNotFoundError(f"SWIFT file not found: {swift_path}")
    
    if references_path:
        references_file = Path(references_path)
        if not references_file.exists():
            raise FileNotFoundError(f"References file not found: {references_path}")
    
    logging.info("File validation passed")


def load_and_prepare_data(baseline_path: str, swift_path: str, max_samples: Optional[int] = None):
    """
    Load and prepare data for evaluation.
    
    Args:
        baseline_path: Path to baseline file
        swift_path: Path to SWIFT file
        max_samples: Maximum samples to load
        
    Returns:
        Tuple of (baseline_data, swift_data, baseline_outputs, swift_outputs)
    """
    logging.info("Loading baseline data...")
    baseline_data = load_jsonl_data(baseline_path, max_samples)
    
    logging.info("Loading SWIFT data...")
    swift_data = load_jsonl_data(swift_path, max_samples)
    
    logging.info("Validating data formats...")
    validate_data_format(baseline_data, "baseline")
    validate_data_format(swift_data, "swift")
    
    logging.info("Filtering valid samples...")
    baseline_data, swift_data = filter_valid_samples(baseline_data, swift_data)
    
    logging.info("Extracting outputs...")
    baseline_outputs = extract_outputs(baseline_data)
    swift_outputs = extract_outputs(swift_data)
    
    logging.info("Aligning outputs...")
    baseline_outputs, swift_outputs = align_outputs(baseline_outputs, swift_outputs)
    
    logging.info(f"Prepared {len(baseline_outputs)} sample pairs for evaluation")
    
    return baseline_data, swift_data, baseline_outputs, swift_outputs


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        setup_logging(args.log_level)
        
        logging.info("Starting SWIFT accuracy evaluation")
        logging.info(f"Baseline: {args.baseline}")
        logging.info(f"SWIFT: {args.swift}")
        logging.info(f"Task: {args.task}")
        
        # Validate files
        validate_files(args.baseline, args.swift, args.references)
        
        # Determine task type
        if args.task == "auto":
            task_name = get_task_name_from_path(args.baseline)
            if task_name == "unknown":
                task_name = get_task_name_from_path(args.swift)
            if task_name == "unknown":
                logging.warning("Could not auto-detect task type, defaulting to 'cnndm'")
                task_name = "cnndm"
        else:
            task_name = args.task
        
        logging.info(f"Using task type: {task_name}")
        
        # Load and prepare data
        baseline_data, swift_data, baseline_outputs, swift_outputs = load_and_prepare_data(
            args.baseline, args.swift, args.max_samples
        )
        
        if len(baseline_outputs) == 0:
            logging.error("No valid samples found for evaluation")
            sys.exit(1)
        
        # Load references if provided
        references = None
        if args.references:
            logging.info("Loading reference data...")
            ref_data = load_jsonl_data(args.references, args.max_samples)
            references = extract_outputs(ref_data)
            logging.info(f"Loaded {len(references)} reference samples")
        
        # Compute metrics
        logging.info("Computing comprehensive metrics...")
        results = compute_comprehensive_metrics(
            baseline_outputs,
            swift_outputs,
            task_name,
            references
        )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results to JSON
        if not args.no_save_json:
            json_path = output_dir / "evaluation_results.json"
            save_results(results, str(json_path))
        
        # Generate and save report
        if not args.no_report:
            logging.info("Generating evaluation report...")
            report = generate_evaluation_report(results, str(output_dir))
            
            # Print report to console if requested
            if args.log_level == "DEBUG":
                print("\n" + "="*80)
                print("EVALUATION REPORT")
                print("="*80)
                print(report)
        
        # Print summary table if requested
        if args.print_summary:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            summary_table = create_comparison_table(results)
            print(summary_table)
            print("")
        
        # Print key metrics to console
        metrics = results.get('metrics', {})
        
        print("\nKEY RESULTS:")
        print("-" * 40)
        
        # Semantic similarity
        if 'semantic_similarity' in metrics:
            sim_data = metrics['semantic_similarity']
            if 'cosine_similarity' in sim_data:
                sim_mean = sim_data['cosine_similarity']['mean']
                print(f"Semantic Similarity: {sim_mean:.4f}")
        
        # Task-specific metrics
        if task_name in ['cnndm', 'summarization']:
            if 'rouge' in metrics and 'relative_scores' in metrics['rouge']:
                rouge_data = metrics['rouge']['relative_scores']
                if 'rouge1' in rouge_data:
                    rouge1_mean = rouge_data['rouge1']['mean']
                    print(f"ROUGE-1 Similarity: {rouge1_mean:.4f}")
            
            if 'bert_score' in metrics and 'relative_scores' in metrics['bert_score']:
                bert_data = metrics['bert_score']['relative_scores']
                if 'f1' in bert_data:
                    bert_f1 = bert_data['f1']['mean']
                    print(f"BERTScore F1: {bert_f1:.4f}")
        
        elif task_name in ['humaneval', 'code_generation']:
            if 'syntax_correctness' in metrics:
                baseline_syntax = metrics['syntax_correctness'].get('baseline', {})
                swift_syntax = metrics['syntax_correctness'].get('swift', {})
                
                baseline_rate = baseline_syntax.get('syntax_correctness_rate', 0)
                swift_rate = swift_syntax.get('syntax_correctness_rate', 0)
                
                print(f"Baseline Syntax Correctness: {baseline_rate:.4f}")
                print(f"SWIFT Syntax Correctness: {swift_rate:.4f}")
                print(f"Difference: {swift_rate - baseline_rate:+.4f}")
        
        # Distribution preservation
        if 'distribution_preservation' in metrics:
            dist_data = metrics['distribution_preservation']
            if 'length_distribution' in dist_data:
                length_data = dist_data['length_distribution']
                ks_significant = length_data.get('ks_test', {}).get('significant', False)
                print(f"Length Distribution Differs: {ks_significant}")
        
        print(f"\nResults saved to: {output_dir}")
        
        logging.info("Evaluation completed successfully")
        
    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()