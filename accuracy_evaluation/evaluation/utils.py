"""
Utility functions for loading and validating SWIFT evaluation data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import jsonlines
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl_data(file_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file with error handling and validation.
    
    Args:
        file_path: Path to the JSONL file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of dictionaries containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is invalid
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.suffix == '.jsonl':
        raise ValueError(f"Expected JSONL file, got: {file_path.suffix}")
    
    data = []
    
    try:
        with jsonlines.open(file_path, 'r') as reader:
            for i, item in enumerate(tqdm(reader, desc=f"Loading {file_path.name}")):
                if max_samples and i >= max_samples:
                    break
                    
                if not isinstance(item, dict):
                    logger.warning(f"Skipping invalid item at line {i+1}: not a dictionary")
                    continue
                    
                data.append(item)
                    
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid JSON format in file {file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {e}")
    
    logger.info(f"Loaded {len(data)} samples from {file_path}")
    return data


def validate_data_format(data: List[Dict[str, Any]], data_type: str = "swift") -> bool:
    """
    Validate that the data has the expected format for SWIFT evaluation.
    
    Args:
        data: List of data dictionaries
        data_type: Type of data ("swift", "baseline")
        
    Returns:
        True if data format is valid
        
    Raises:
        ValueError: If data format is invalid
    """
    if not data:
        raise ValueError("Data list is empty")
    
    required_fields = ["choices", "model_id"]
    choice_fields = ["turns", "new_tokens", "wall_time"]
    
    for i, item in enumerate(data):
        # Skip metadata entries (they don't have choices)
        if "Mean accepted tokens" in item or "Token acceptance rate" in item:
            continue
            
        # Check required top-level fields
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Missing required field '{field}' in item {i}")
        
        # Check choices structure
        if not isinstance(item["choices"], list) or len(item["choices"]) == 0:
            raise ValueError(f"Invalid 'choices' field in item {i}")
        
        choice = item["choices"][0]
        for field in choice_fields:
            if field not in choice:
                raise ValueError(f"Missing required choice field '{field}' in item {i}")
    
    logger.info(f"Data format validation passed for {len(data)} items")
    return True


def extract_outputs(data: List[Dict[str, Any]]) -> List[str]:
    """
    Extract the generated text outputs from SWIFT/baseline data.
    
    Args:
        data: List of data dictionaries from JSONL
        
    Returns:
        List of generated text outputs
    """
    outputs = []
    
    for item in data:
        # Skip metadata entries
        if "Mean accepted tokens" in item or "Token acceptance rate" in item:
            continue
            
        if "choices" in item and len(item["choices"]) > 0:
            choice = item["choices"][0]
            if "turns" in choice:
                output = choice["turns"]
                # Clean up the output text
                output = output.strip()
                outputs.append(output)
    
    logger.info(f"Extracted {len(outputs)} outputs from data")
    return outputs


def extract_reference_summaries(task_name: str, data_indices: List[int]) -> List[str]:
    """
    Extract reference summaries for evaluation (for tasks where references are needed).
    
    Args:
        task_name: Name of the task ("cnndm", "humaneval", etc.)
        data_indices: Indices of the data samples to get references for
        
    Returns:
        List of reference texts
        
    Note:
        This function may need to be adapted based on how reference data is stored
    """
    references = []
    
    if task_name == "cnndm":
        # For CNN/DailyMail, we would need access to the original dataset
        # This is a placeholder - in practice, you'd load the dataset
        logger.warning("Reference summary extraction not implemented for cnndm")
        references = ["" for _ in data_indices]  # Placeholder
        
    elif task_name == "humaneval":
        # For HumanEval, references would be the expected function behavior
        # This is typically handled by test cases rather than reference text
        logger.warning("Reference extraction not applicable for humaneval")
        references = ["" for _ in data_indices]  # Placeholder
        
    else:
        logger.warning(f"Unknown task name: {task_name}")
        references = ["" for _ in data_indices]
    
    return references


def align_outputs(baseline_outputs: List[str], swift_outputs: List[str]) -> Tuple[List[str], List[str]]:
    """
    Align baseline and SWIFT outputs for comparison.
    
    Args:
        baseline_outputs: List of baseline outputs
        swift_outputs: List of SWIFT outputs
        
    Returns:
        Tuple of (aligned_baseline, aligned_swift)
        
    Raises:
        ValueError: If the outputs cannot be aligned
    """
    if len(baseline_outputs) != len(swift_outputs):
        min_len = min(len(baseline_outputs), len(swift_outputs))
        logger.warning(
            f"Output lengths don't match: baseline={len(baseline_outputs)}, "
            f"swift={len(swift_outputs)}. Using first {min_len} samples."
        )
        return baseline_outputs[:min_len], swift_outputs[:min_len]
    
    return baseline_outputs, swift_outputs


def get_task_name_from_path(file_path: str) -> str:
    """
    Attempt to extract task name from file path.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        Detected task name or "unknown"
    """
    path_lower = file_path.lower()
    
    if "cnndm" in path_lower or "cnn" in path_lower:
        return "cnndm"
    elif "humaneval" in path_lower or "human_eval" in path_lower:
        return "humaneval"
    else:
        return "unknown"


def filter_valid_samples(baseline_data: List[Dict], swift_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter out invalid samples from both datasets to ensure fair comparison.
    
    Args:
        baseline_data: Baseline data samples
        swift_data: SWIFT data samples
        
    Returns:
        Tuple of (filtered_baseline, filtered_swift)
    """
    valid_baseline = []
    valid_swift = []
    
    min_len = min(len(baseline_data), len(swift_data))
    
    for i in range(min_len):
        baseline_item = baseline_data[i]
        swift_item = swift_data[i]
        
        # Skip metadata entries
        if ("Mean accepted tokens" in baseline_item or "Token acceptance rate" in baseline_item or
            "Mean accepted tokens" in swift_item or "Token acceptance rate" in swift_item):
            continue
        
        # Check if both have valid outputs
        baseline_valid = ("choices" in baseline_item and 
                         len(baseline_item["choices"]) > 0 and
                         "turns" in baseline_item["choices"][0])
        
        swift_valid = ("choices" in swift_item and 
                      len(swift_item["choices"]) > 0 and
                      "turns" in swift_item["choices"][0])
        
        if baseline_valid and swift_valid:
            valid_baseline.append(baseline_item)
            valid_swift.append(swift_item)
    
    logger.info(f"Filtered to {len(valid_baseline)} valid sample pairs")
    return valid_baseline, valid_swift