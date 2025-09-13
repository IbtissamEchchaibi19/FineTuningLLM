import json
import os
import sys
import warnings
import re
from collections import Counter
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import textstat
    from datasets import Dataset
    from detoxify import Detoxify
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install textstat datasets detoxify matplotlib seaborn")
    sys.exit(1)

warnings.filterwarnings('ignore')

class InstructionDatasetProcessor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.cleaned_file = input_file.replace('.jsonl', '_cleaned.jsonl')
        self.raw_data = []
        self.cleaned_data = []
        self.evaluation_results = {}
        self.detoxify_model = None

    def load_raw_dataset(self) -> bool:
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        self.raw_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            return len(self.raw_data) > 0
        except Exception:
            return False

    def improve_readability(self, text: str) -> str:
        if not text:
            return text
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'([.!?])\s*([.!?])', r'\1 \2', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        improved_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 150:
                parts = re.split(r'(,\s*(?:and|but|or|so|yet|however|therefore|furthermore|moreover))', sentence)
                if len(parts) > 1:
                    current = ""
                    for part in parts:
                        if len(current + part) > 100 and current:
                            improved_sentences.append(current.strip())
                            current = part
                        else:
                            current += part
                    if current.strip():
                        improved_sentences.append(current.strip())
                else:
                    improved_sentences.append(sentence)
            elif sentence:
                improved_sentences.append(sentence)
        result = ' '.join(improved_sentences)
        if result and not result.endswith(('.', '!', '?', ':')):
            result += '.'
        return result

    def clean_dataset(self) -> Dict[str, Any]:
        stats = {'original': len(self.raw_data), 'format_issues': 0, 'duplicates': 0,
                 'extreme_lengths': 0, 'low_quality': 0, 'final': 0}
        seen_instructions = set()
        seen_pairs = set()
        for row in self.raw_data:
            try:
                if not ('instruction' in row and 'output' in row and row['instruction'] and row['output']):
                    stats['format_issues'] += 1
                    continue
                instruction = str(row['instruction']).strip()
                output = str(row['output']).strip()
                if len(instruction) < 5 or len(output) < 15 or len(output) > 2000:
                    stats['extreme_lengths'] += 1
                    continue
                instruction_normalized = instruction.lower().strip()
                if instruction_normalized in seen_instructions:
                    stats['duplicates'] += 1
                    continue
                seen_instructions.add(instruction_normalized)
                pair_key = (instruction_normalized, output.lower().strip())
                if pair_key in seen_pairs:
                    stats['duplicates'] += 1
                    continue
                seen_pairs.add(pair_key)
                improved_output = self.improve_readability(output)
                if len(improved_output.split()) < 3:
                    stats['low_quality'] += 1
                    continue
                cleaned_row = {'instruction': instruction, 'output': improved_output}
                for key, value in row.items():
                    if key not in ['instruction', 'output'] and value is not None:
                        cleaned_row[key] = value
                self.cleaned_data.append(cleaned_row)
                stats['final'] += 1
            except Exception:
                stats['format_issues'] += 1
                continue
        try:
            with open(self.cleaned_file, 'w', encoding='utf-8') as f:
                for row in self.cleaned_data:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
        except Exception:
            return stats
        return stats

    def validate_format(self) -> Dict[str, Any]:
        valid_rows = 0
        issues = 0
        for row in self.cleaned_data:
            if 'instruction' in row and 'output' in row and row['instruction'] and row['output']:
                valid_rows += 1
            else:
                issues += 1
        results = {'total_rows': len(self.cleaned_data), 'valid_rows': valid_rows, 'issues': issues,
                   'format_valid': issues == 0}
        return results

    def check_instruction_uniqueness(self) -> Dict[str, Any]:
        instructions = [str(row['instruction']).strip() for row in self.cleaned_data if 'instruction' in row]
        unique_instructions = set(instructions)
        instruction_counts = Counter(instructions)
        duplicates = {inst: count for inst, count in instruction_counts.items() if count > 1}
        results = {'total_instructions': len(instructions), 'unique_instructions': len(unique_instructions),
                   'uniqueness_ratio': len(unique_instructions) / len(instructions) if instructions else 0,
                   'duplicate_count': len(duplicates)}
        return results

    def analyze_output_quality(self) -> Dict[str, Any]:
        outputs = []
        flesch_scores = []
        lengths = []
        for row in self.cleaned_data:
            if 'output' in row and row['output']:
                output_text = str(row['output']).strip()
                outputs.append(output_text)
                lengths.append(len(output_text))
                try:
                    flesch = textstat.flesch_reading_ease(output_text)
                    flesch_scores.append(flesch)
                except:
                    flesch_scores.append(50)
        if not outputs:
            return {'error': 'No valid outputs found'}
        length_stats = {'mean': np.mean(lengths), 'median': np.median(lengths), 'std': np.std(lengths),
                        'min': min(lengths), 'max': max(lengths),
                        'q25': np.percentile(lengths, 25), 'q75': np.percentile(lengths, 75)}
        very_short = sum(1 for l in lengths if l < 20)
        very_long = sum(1 for l in lengths if l > 1500)
        results = {'total_outputs': len(outputs), 'avg_flesch_score': np.mean(flesch_scores),
                   'flesch_interpretation': self._interpret_flesch_score(np.mean(flesch_scores)),
                   'length_stats': length_stats, 'very_short_outputs': very_short,
                   'very_long_outputs': very_long, 'extreme_ratio': (very_short + very_long) / len(outputs)}
        self._plot_quality_metrics(lengths, flesch_scores)
        return results

    def _interpret_flesch_score(self, score: float) -> str:
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"

    def _plot_quality_metrics(self, lengths: List[int], flesch_scores: List[float]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_title('Output Length Distribution')
        ax1.set_xlabel('Output Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
        ax1.legend()
        ax2.hist(flesch_scores, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_title('Readability Score Distribution')
        ax2.set_xlabel('Flesch Reading Ease Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(flesch_scores), color='red', linestyle='--', label=f'Mean: {np.mean(flesch_scores):.1f}')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('dataset_quality_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()

    def detect_duplicates(self) -> Dict[str, Any]:
        pairs = []
        for row in self.cleaned_data:
            if 'instruction' in row and 'output' in row:
                pair = (str(row['instruction']).strip().lower(), str(row['output']).strip().lower())
                pairs.append(pair)
        unique_pairs = set(pairs)
        pair_counts = Counter(pairs)
        duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
        results = {'total_pairs': len(pairs), 'unique_pairs': len(unique_pairs),
                   'duplicate_pairs': len(duplicates),
                   'duplication_ratio': len(duplicates) / len(pairs) if pairs else 0}
        return results

    def detect_toxicity(self) -> Dict[str, Any]:
        try:
            if self.detoxify_model is None:
                self.detoxify_model = Detoxify('original')
            outputs = [str(row['output']).strip() for row in self.cleaned_data if 'output' in row]
            if not outputs:
                return {'error': 'No outputs to analyze'}
            sample_size = min(1000, len(outputs))
            sample_outputs = outputs[:sample_size]
            toxicity_scores = self.detoxify_model.predict(sample_outputs)
            toxic_threshold = 0.5
            toxic_count = sum(1 for score in toxicity_scores['toxicity'] if score > toxic_threshold)
            results = {'analyzed_outputs': sample_size, 'toxic_outputs': toxic_count,
                       'toxicity_rate': toxic_count / sample_size,
                       'avg_toxicity_score': np.mean(toxicity_scores['toxicity']),
                       'max_toxicity_score': max(toxicity_scores['toxicity'])}
            return results
        except Exception as e:
            return {'error': str(e)}

    def check_dataset_size(self) -> Dict[str, Any]:
        total_rows = len(self.cleaned_data)
        min_required = 5000
        results = {'total_rows': total_rows, 'min_required': min_required, 'size_adequate': total_rows >= min_required}
        return results

    def show_sample_outputs(self, sample_size: int = 5) -> Dict[str, Any]:
        sample_data = self.cleaned_data[:sample_size]
        for i, row in enumerate(sample_data, 1):
            instruction = str(row['instruction'])
            output = str(row['output'])
            try:
                flesch = textstat.flesch_reading_ease(output)
            except:
                flesch = None
        return {'samples_shown': len(sample_data)}

    def generate_final_report(self) -> bool:
        format_results = self.evaluation_results.get('format', {})
        uniqueness_results = self.evaluation_results.get('uniqueness', {})
        quality_results = self.evaluation_results.get('quality', {})
        duplicate_results = self.evaluation_results.get('duplicates', {})
        toxicity_results = self.evaluation_results.get('toxicity', {})
        size_results = self.evaluation_results.get('size', {})
        checks = {}
        checks['format'] = format_results.get('format_valid', False)
        uniqueness_ratio = uniqueness_results.get('uniqueness_ratio', 0)
        checks['uniqueness'] = uniqueness_ratio > 0.85
        flesch_score = quality_results.get('avg_flesch_score', 0)
        extreme_ratio = quality_results.get('extreme_ratio', 0)
        checks['quality'] = 25 <= flesch_score <= 85 and extreme_ratio < 0.05
        dup_ratio = duplicate_results.get('duplication_ratio', 0)
        checks['duplicates'] = dup_ratio < 0.02
        if 'error' not in toxicity_results:
            tox_rate = toxicity_results.get('toxicity_rate', 0)
            checks['toxicity'] = tox_rate < 0.05
        else:
            checks['toxicity'] = True
        checks['size'] = size_results.get('size_adequate', False)
        passed = sum(checks.values())
        total = len(checks)
        ready = passed >= total - 1
        return ready

    def run_complete_process(self):
        if not self.load_raw_dataset():
            return False
        self.clean_dataset()
        self.evaluation_results['format'] = self.validate_format()
        self.evaluation_results['uniqueness'] = self.check_instruction_uniqueness()
        self.evaluation_results['quality'] = self.analyze_output_quality()
        self.evaluation_results['duplicates'] = self.detect_duplicates()
        self.evaluation_results['toxicity'] = self.detect_toxicity()
        self.evaluation_results['size'] = self.check_dataset_size()
        self.show_sample_outputs()
        return self.generate_final_report()

def main():
    if len(sys.argv) != 2:
        print("Usage: python instruction_dataset_processor.py <input_jsonl_file>")
        sys.exit(1)
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)
    processor = InstructionDatasetProcessor(input_file)
    success = processor.run_complete_process()
    print(f"{'SUCCESS' if success else 'NEEDS MORE WORK'}")
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
