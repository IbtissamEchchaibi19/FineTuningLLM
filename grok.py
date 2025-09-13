import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datasets import load_dataset
from textstat import flesch_reading_ease
from detoxify import Detoxify
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl_dataset(file_path: str) -> List[Dict]:
    """Load JSONL dataset and return list of dictionaries."""
    try:
        dataset = load_dataset('json', data_files=file_path, split='train')
        return [dict(item) for item in dataset]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def validate_format(dataset: List[Dict]) -> Tuple[bool, List[str]]:
    """Check if all rows have non-empty 'instruction' and 'output' fields."""
    errors = []
    for i, row in enumerate(dataset):
        if 'instruction' not in row or not row['instruction'].strip():
            errors.append(f"Row {i+1}: Missing or empty 'instruction' field")
        if 'output' not in row or not row['output'].strip():
            errors.append(f"Row {i+1}: Missing or empty 'output' field")
    return len(errors) == 0, errors

def check_instruction_uniqueness(dataset: List[Dict]) -> Tuple[float, int]:
    """Calculate percentage of unique instructions and total count."""
    instructions = [row['instruction'] for row in dataset]
    unique_instructions = len(set(instructions))
    total_instructions = len(instructions)
    uniqueness_ratio = (unique_instructions / total_instructions) * 100 if total_instructions > 0 else 0
    return uniqueness_ratio, unique_instructions

def analyze_output_quality(dataset: List[Dict]) -> Tuple[float, Dict, List[float]]:
    """Calculate Flesch Reading Ease and output length distribution."""
    flesch_scores = []
    lengths = []
    
    for row in dataset:
        output = row['output']
        try:
            score = flesch_reading_ease(output)
            flesch_scores.append(score)
        except Exception as e:
            logger.warning(f"Error calculating Flesch score for output in row: {e}")
            flesch_scores.append(None)
        
        length = len(output.split())
        lengths.append(length)
    
    avg_flesch = np.mean([s for s in flesch_scores if s is not None]) if any(s is not None for s in flesch_scores) else 0
    length_stats = {
        'mean': np.mean(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths),
        'short_count': sum(1 for l in lengths if l < 10),  # Arbitrarily short: <10 words
        'long_count': sum(1 for l in lengths if l > 500)   # Arbitrarily long: >500 words
    }
    
    return avg_flesch, length_stats, lengths

def detect_duplicates(dataset: List[Dict]) -> Tuple[int, List[Tuple[int, int]]]:
    """Identify duplicate (instruction, output) pairs."""
    seen = {}
    duplicates = []
    for i, row in enumerate(dataset):
        key = (row['instruction'], row['output'])
        if key in seen:
            duplicates.append((seen[key] + 1, i + 1))
        else:
            seen[key] = i
    return len(duplicates), duplicates

def check_toxicity(dataset: List[Dict], threshold: float = 0.7) -> Tuple[int, List[Dict]]:
    """Flag toxic outputs using Detoxify."""
    try:
        model = Detoxify('original')
    except Exception as e:
        logger.warning(f"Failed to load Detoxify model: {e}. Skipping toxicity check.")
        return 0, []
    
    toxic_outputs = []
    for i, row in enumerate(dataset):
        try:
            results = model.predict(row['output'])
            if results['toxicity'] > threshold:
                toxic_outputs.append({
                    'row': i + 1,
                    'output': row['output'][:100],  # Truncate for brevity
                    'toxicity_score': results['toxicity']
                })
        except Exception as e:
            logger.warning(f"Error processing row {i+1} for toxicity: {e}")
    
    return len(toxic_outputs), toxic_outputs

def plot_length_distribution(lengths: List[float], output_path: str = 'length_distribution.png'):
    """Plot output length distribution and save to file."""
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, edgecolor='black')
    plt.title('Output Length Distribution (Words)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Length distribution plot saved to {output_path}")

def evaluate_dataset(file_path: str, min_rows: int = 5000) -> Dict:
    """Main function to evaluate dataset suitability for fine-tuning."""
    logger.info("Starting dataset evaluation...")
    
    # Load dataset
    dataset = load_jsonl_dataset(file_path)
    total_rows = len(dataset)
    logger.info(f"Loaded dataset with {total_rows} rows")
    
    # Initialize results
    results = {
        'format_valid': False,
        'format_errors': [],
        'uniqueness_ratio': 0.0,
        'unique_instructions': 0,
        'avg_flesch_score': 0.0,
        'length_stats': {},
        'duplicate_count': 0,
        'duplicates': [],
        'toxic_count': 0,
        'toxic_outputs': [],
        'sufficient_size': False,
        'ready_for_finetuning': False,
        'issues': []
    }
    
    # 1. Format validation
    results['format_valid'], results['format_errors'] = validate_format(dataset)
    if not results['format_valid']:
        results['issues'].append(f"Format issues found: {len(results['format_errors'])} rows with missing or empty fields")
    
    # 2. Instruction uniqueness
    results['uniqueness_ratio'], results['unique_instructions'] = check_instruction_uniqueness(dataset)
    if results['uniqueness_ratio'] < 50:
        results['issues'].append(f"Low instruction uniqueness: {results['uniqueness_ratio']:.2f}% unique instructions")
    
    # 3. Output quality
    results['avg_flesch_score'], results['length_stats'], lengths = analyze_output_quality(dataset)
    if results['avg_flesch_score'] < 30:
        results['issues'].append(f"Low average Flesch Reading Ease score: {results['avg_flesch_score']:.2f}")
    if results['length_stats']['short_count'] > total_rows * 0.1:
        results['issues'].append(f"Too many short outputs: {results['length_stats']['short_count']} outputs <10 words")
    if results['length_stats']['long_count'] > total_rows * 0.05:
        results['issues'].append(f"Too many long outputs: {results['length_stats']['long_count']} outputs >500 words")
    
    # Plot length distribution
    plot_length_distribution(lengths)
    
    # 4. Duplicate detection
    results['duplicate_count'], results['duplicates'] = detect_duplicates(dataset)
    if results['duplicate_count'] > total_rows * 0.05:
        results['issues'].append(f"Too many duplicates: {results['duplicate_count']} duplicate pairs")
    
    # 5. Toxicity detection
    results['toxic_count'], results['toxic_outputs'] = check_toxicity(dataset)
    if results['toxic_count'] > 0:
        results['issues'].append(f"Toxic outputs detected: {results['toxic_count']} rows with toxicity score >0.7")
    
    # 6. Dataset size
    clean_rows = total_rows - results['toxic_count'] - results['duplicate_count'] - len(results['format_errors'])
    results['sufficient_size'] = clean_rows >= min_rows
    if not results['sufficient_size']:
        results['issues'].append(f"Insufficient clean rows: {clean_rows} clean rows (required: {min_rows})")
    
    # 7. Final recommendation
    results['ready_for_finetuning'] = (
        results['format_valid'] and
        results['uniqueness_ratio'] >= 50 and
        results['avg_flesch_score'] >= 30 and
        results['length_stats']['short_count'] <= total_rows * 0.1 and
        results['length_stats']['long_count'] <= total_rows * 0.05 and
        results['duplicate_count'] <= total_rows * 0.05 and
        results['toxic_count'] == 0 and
        results['sufficient_size']
    )
    
    return results

def print_summary(results: Dict):
    """Print evaluation summary and recommendation."""
    print("\n=== Dataset Evaluation Summary ===")
    print(f"Total Rows: {results['unique_instructions'] + results['duplicate_count']}")
    print(f"Format Validation: {'✅ Passed' if results['format_valid'] else '❌ Failed'}")
    if results['format_errors']:
        print(f"  - Errors: {len(results['format_errors'])} (e.g., {results['format_errors'][:2]})")
    
    print(f"Instruction Uniqueness: {results['uniqueness_ratio']:.2f}% ({results['unique_instructions']} unique)")
    print(f"Average Flesch Reading Ease: {results['avg_flesch_score']:.2f}")
    print(f"Output Length Stats:")
    print(f"  - Mean: {results['length_stats']['mean']:.2f} words")
    print(f"  - Min: {results['length_stats']['min']} words")
    print(f"  - Max: {results['length_stats']['max']} words")
    print(f"  - Short outputs (<10 words): {results['length_stats']['short_count']}")
    print(f"  - Long outputs (>500 words): {results['length_stats']['long_count']}")
    print(f"Duplicates: {results['duplicate_count']} pairs")
    if results['duplicates']:
        print(f"  - Sample duplicates (row pairs): {results['duplicates'][:2]}")
    print(f"Toxic Outputs: {results['toxic_count']}")
    if results['toxic_outputs']:
        print(f"  - Sample toxic outputs: {results['toxic_outputs'][:2]}")
    print(f"Sufficient Size (≥5000 clean rows): {'✅ Yes' if results['sufficient_size'] else '❌ No'}")
    
    print("\nIssues Found:")
    if results['issues']:
        for issue in results['issues']:
            print(f"  - {issue}")
    else:
        print("  - None")
    
    print("\nRecommendation:")
    print("✅ Good for fine-tuning" if results['ready_for_finetuning'] else "❌ Not ready yet")

if __name__ == "__main__":
    # Replace with your JSONL file path
    file_path = "ready_data.jsonl"
    try:
        results = evaluate_dataset(file_path)
        print_summary(results)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print("❌ Evaluation failed. Check logs for details.")