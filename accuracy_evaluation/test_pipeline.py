#!/usr/bin/env python3
"""
Test script for SWIFT accuracy evaluation pipeline.

This script creates sample data and tests the evaluation pipeline to ensure
all components are working correctly.
"""

import json
import tempfile
import os
import sys
from pathlib import Path
import logging
import subprocess

# Add the evaluation package to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from evaluation import (
        load_jsonl_data,
        validate_data_format,
        extract_outputs,
        compute_comprehensive_metrics
    )
except ImportError as e:
    print(f"Error importing evaluation modules: {e}")
    print("Please ensure all requirements are installed: pip install -r requirements.txt")
    sys.exit(1)


def create_sample_cnndm_data(num_samples: int = 5) -> tuple:
    """
    Create sample CNN/DailyMail data for testing.
    
    Args:
        num_samples: Number of sample entries to create
        
    Returns:
        Tuple of (baseline_data, swift_data)
    """
    baseline_data = []
    swift_data = []
    
    sample_articles = [
        "The new research shows that artificial intelligence can significantly improve healthcare outcomes.",
        "Climate change continues to be one of the most pressing challenges of our time.",
        "The latest technology developments in quantum computing promise revolutionary advances.",
        "Economic indicators suggest a positive trend in the global market recovery.",
        "Space exploration missions are revealing new insights about our universe."
    ]
    
    baseline_summaries = [
        "AI improves healthcare outcomes according to new research.",
        "Climate change remains a major global challenge.",
        "Quantum computing technology shows revolutionary potential.", 
        "Global market recovery shows positive economic trends.",
        "Space missions provide new universal insights."
    ]
    
    swift_summaries = [
        "New research demonstrates AI's significant impact on healthcare outcomes.",
        "Climate change continues as one of today's most pressing challenges.",
        "Revolutionary advances promised by quantum computing developments.",
        "Positive global market recovery trends indicated by economic data.",
        "New universal insights revealed through space exploration missions."
    ]
    
    for i in range(num_samples):
        idx = i % len(sample_articles)
        
        # Baseline entry
        baseline_entry = {
            "model_id": "llama-2-13b-vanilla",
            "choices": [{
                "turns": baseline_summaries[idx],
                "decoding_steps": [25],
                "new_tokens": [len(baseline_summaries[idx].split())],
                "wall_time": [2.5],
                "accept_lengths": [1] * len(baseline_summaries[idx].split()),
                "acceptance_rate": 0.0
            }],
            "tstamp": 1699123456.789 + i
        }
        
        # SWIFT entry
        swift_entry = {
            "model_id": "llama-2-13b-swift",
            "choices": [{
                "turns": swift_summaries[idx],
                "decoding_steps": [20],
                "new_tokens": [len(swift_summaries[idx].split())],
                "wall_time": [1.8],
                "accept_lengths": [1, 2, 1, 2] * (len(swift_summaries[idx].split()) // 4 + 1),
                "acceptance_rate": 0.65
            }],
            "tstamp": 1699123456.789 + i
        }
        
        baseline_data.append(baseline_entry)
        swift_data.append(swift_entry)
    
    return baseline_data, swift_data


def create_sample_humaneval_data(num_samples: int = 5) -> tuple:
    """
    Create sample HumanEval data for testing.
    
    Args:
        num_samples: Number of sample entries to create
        
    Returns:
        Tuple of (baseline_data, swift_data)
    """
    baseline_data = []
    swift_data = []
    
    baseline_code = [
        "def add_numbers(a, b):\n    return a + b",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def reverse_string(s):\n    return s[::-1]"
    ]
    
    swift_code = [
        "def add_numbers(a, b):\n    # Add two numbers\n    return a + b",
        "def factorial(n):\n    # Calculate factorial recursively\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "def fibonacci(n):\n    # Generate fibonacci number\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "def is_prime(n):\n    # Check if number is prime\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def reverse_string(s):\n    # Reverse a string\n    return s[::-1]"
    ]
    
    for i in range(num_samples):
        idx = i % len(baseline_code)
        
        # Baseline entry
        baseline_entry = {
            "model_id": "codellama-13b-vanilla",
            "choices": [{
                "turns": baseline_code[idx],
                "decoding_steps": [30],
                "new_tokens": [len(baseline_code[idx].split())],
                "wall_time": [3.2],
                "accept_lengths": [1] * len(baseline_code[idx].split()),
                "acceptance_rate": 0.0
            }],
            "tstamp": 1699123456.789 + i
        }
        
        # SWIFT entry
        swift_entry = {
            "model_id": "codellama-13b-swift", 
            "choices": [{
                "turns": swift_code[idx],
                "decoding_steps": [25],
                "new_tokens": [len(swift_code[idx].split())],
                "wall_time": [2.1],
                "accept_lengths": [1, 2, 1, 3] * (len(swift_code[idx].split()) // 4 + 1),
                "acceptance_rate": 0.72
            }],
            "tstamp": 1699123456.789 + i
        }
        
        baseline_data.append(baseline_entry)
        swift_data.append(swift_entry)
    
    return baseline_data, swift_data


def save_jsonl(data: list, filepath: str) -> None:
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def test_data_loading():
    """Test data loading and validation functions."""
    print("Testing data loading and validation...")
    
    # Create temporary files with sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        baseline_data, swift_data = create_sample_cnndm_data(3)
        
        baseline_file = os.path.join(temp_dir, "baseline.jsonl")
        swift_file = os.path.join(temp_dir, "swift.jsonl")
        
        save_jsonl(baseline_data, baseline_file)
        save_jsonl(swift_data, swift_file)
        
        # Test loading
        loaded_baseline = load_jsonl_data(baseline_file)
        loaded_swift = load_jsonl_data(swift_file)
        
        assert len(loaded_baseline) == 3, f"Expected 3 baseline samples, got {len(loaded_baseline)}"
        assert len(loaded_swift) == 3, f"Expected 3 SWIFT samples, got {len(loaded_swift)}"
        
        # Test validation
        validate_data_format(loaded_baseline, "baseline")
        validate_data_format(loaded_swift, "swift")
        
        # Test output extraction
        baseline_outputs = extract_outputs(loaded_baseline)
        swift_outputs = extract_outputs(loaded_swift)
        
        assert len(baseline_outputs) == 3, f"Expected 3 baseline outputs, got {len(baseline_outputs)}"
        assert len(swift_outputs) == 3, f"Expected 3 SWIFT outputs, got {len(swift_outputs)}"
        
        print("Data loading and validation tests passed")


def test_metrics_computation():
    """Test metrics computation functions."""
    print("Testing metrics computation...")
    
    # Test with CNN/DM data
    baseline_data, swift_data = create_sample_cnndm_data(5)
    baseline_outputs = extract_outputs(baseline_data)
    swift_outputs = extract_outputs(swift_data)
    
    results = compute_comprehensive_metrics(
        baseline_outputs, swift_outputs, "cnndm"
    )
    
    assert "metrics" in results, "Results should contain metrics"
    assert "semantic_similarity" in results["metrics"], "Should compute semantic similarity"
    
    print("CNN/DM metrics computation test passed")
    
    # Test with HumanEval data
    baseline_data, swift_data = create_sample_humaneval_data(5)
    baseline_outputs = extract_outputs(baseline_data)
    swift_outputs = extract_outputs(swift_data)
    
    results = compute_comprehensive_metrics(
        baseline_outputs, swift_outputs, "humaneval"
    )
    
    assert "metrics" in results, "Results should contain metrics"
    assert "syntax_correctness" in results["metrics"], "Should compute syntax correctness"
    
    print("HumanEval metrics computation test passed")


def test_cli_interface():
    """Test the CLI interface with sample data."""
    print("Testing CLI interface...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data files
        baseline_data, swift_data = create_sample_cnndm_data(3)
        
        baseline_file = os.path.join(temp_dir, "cnndm_baseline.jsonl")
        swift_file = os.path.join(temp_dir, "cnndm_swift.jsonl")
        output_dir = os.path.join(temp_dir, "results")
        
        save_jsonl(baseline_data, baseline_file)
        save_jsonl(swift_data, swift_file)
        
        # Test CLI execution
        script_path = Path(__file__).parent / "evaluate_accuracy.py"
        cmd = [
            sys.executable, str(script_path),
            "--baseline", baseline_file,
            "--swift", swift_file,
            "--task", "cnndm",
            "--output-dir", output_dir,
            "--log-level", "WARNING"  # Reduce log noise
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"CLI test failed with return code {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
            
            # Check if output files were created
            results_file = os.path.join(output_dir, "evaluation_results.json")
            if not os.path.exists(results_file):
                print("CLI test failed: results file not created")
                return False
                
            print("CLI interface test passed")
            return True
            
        except subprocess.TimeoutExpired:
            print("CLI test failed: timeout")
            return False
        except Exception as e:
            print(f"CLI test failed: {e}")
            return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SWIFT ACCURACY EVALUATION PIPELINE TESTS")
    print("=" * 60)
    
    try:
        # Setup logging for tests
        logging.basicConfig(level=logging.WARNING)
        
        # Run tests
        test_data_loading()
        test_metrics_computation()
        
        # CLI test (may take longer)
        print("Testing CLI interface (this may take a moment)...")
        cli_success = test_cli_interface()
        
        print("\n" + "=" * 60)
        if cli_success:
            print("ALL TESTS PASSED!")
            print("The SWIFT accuracy evaluation pipeline is ready to use.")
        else:
            print("CLI TEST FAILED")
            print("Basic functionality works, but CLI interface has issues.")
        print("=" * 60)
        
        return cli_success
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)