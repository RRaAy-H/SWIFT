#!/usr/bin/env python3
"""
Example usage script for SWIFT accuracy evaluation pipeline.

This script demonstrates how to use the evaluation pipeline with
existing SWIFT evaluation outputs.
"""

import os
import sys
import argparse
from pathlib import Path

def find_evaluation_files(outputs_dir: str, model_name: str, task_name: str):
    """
    Find baseline and SWIFT evaluation files in the outputs directory.
    
    Args:
        outputs_dir: Path to outputs directory
        model_name: Name of the model (e.g., "llama-2-13b")
        task_name: Name of the task (e.g., "cnndm", "humaneval")
        
    Returns:
        Tuple of (baseline_file, swift_file) or (None, None) if not found
    """
    # Expected path structure: outputs/task_name/model_answer/model_name/
    task_dir = Path(outputs_dir) / task_name / "model_answer" / model_name
    
    if not task_dir.exists():
        print(f"Task directory not found: {task_dir}")
        return None, None
    
    # Look for baseline and SWIFT files
    baseline_file = None
    swift_file = None
    
    for file in task_dir.glob("*.jsonl"):
        if "vanilla" in file.name.lower() or "baseline" in file.name.lower():
            baseline_file = file
        elif "swift" in file.name.lower():
            swift_file = file
    
    return baseline_file, swift_file


def main():
    parser = argparse.ArgumentParser(
        description="Example usage of SWIFT accuracy evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate CNN/DM summarization results
  python example_usage.py --outputs-dir ../outputs --model llama-2-13b --task cnndm_100

  # Evaluate HumanEval code generation results  
  python example_usage.py --outputs-dir ../outputs --model codellama-13b --task humaneval_100
        """
    )
    
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="../outputs",
        help="Path to SWIFT outputs directory"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., llama-2-13b, codellama-13b)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., cnndm_100, humaneval_100)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: results_{model}_{task})"
    )
    
    args = parser.parse_args()
    
    # Find evaluation files
    print(f"Looking for evaluation files...")
    print(f"  Outputs dir: {args.outputs_dir}")
    print(f"  Model: {args.model}")
    print(f"  Task: {args.task}")
    
    baseline_file, swift_file = find_evaluation_files(
        args.outputs_dir, args.model, args.task
    )
    
    if not baseline_file or not swift_file:
        print("\nCould not find both baseline and SWIFT files!")
        print("Expected directory structure:")
        print(f"  {args.outputs_dir}/{args.task}/model_answer/{args.model}/")
        print("  ├── {model}-vanilla.jsonl  (or {model}-baseline.jsonl)")
        print("  └── {model}-swift.jsonl")
        print("\nAvailable files:")
        
        task_dir = Path(args.outputs_dir) / args.task / "model_answer" / args.model
        if task_dir.exists():
            for file in task_dir.glob("*.jsonl"):
                print(f"    {file.name}")
        else:
            print(f"    Directory not found: {task_dir}")
        
        sys.exit(1)
    
    print(f"✓ Found baseline file: {baseline_file}")
    print(f"✓ Found SWIFT file: {swift_file}")
    
    # Determine task type
    task_type = "auto"
    if "cnndm" in args.task.lower() or "cnn" in args.task.lower():
        task_type = "cnndm"
    elif "humaneval" in args.task.lower() or "human_eval" in args.task.lower():
        task_type = "humaneval"
    
    print(f"Detected task type: {task_type}")
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results_{args.model}_{args.task}"
    
    # Build evaluation command
    evaluation_script = Path(__file__).parent / "evaluate_accuracy.py"
    
    cmd = [
        sys.executable, str(evaluation_script),
        "--baseline", str(baseline_file),
        "--swift", str(swift_file),
        "--task", task_type,
        "--output-dir", output_dir,
        "--print-summary"
    ]
    
    print(f"\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the evaluation
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print(f"Evaluation completed successfully!")
        print(f"Results saved to: {output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()