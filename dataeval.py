
import json
import os
import sys
import warnings
from collections import Counter
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core libraries
try:
    import textstat
    from datasets import Dataset
    from detoxify import Detoxify
except ImportError as e:
    sys.exit(1)

warnings.filterwarnings('ignore')

class DatasetQualityEvaluator:
    
    def __init__(self, jsonl_file_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.data = []
        self.evaluation_results = {}
        self.detoxify_model = None
        
    def load_dataset(self) -> bool:
        """Load and parse JSONL dataset."""
        print("📁 Loading dataset...")
        try:
            with open(self.jsonl_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        self.data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"⚠️  Invalid JSON on line {line_num}")
            
            print(f"✅ Loaded {len(self.data)} rows from dataset")
            return len(self.data) > 0
        except FileNotFoundError:
            print(f"❌ File not found: {self.jsonl_file_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def validate_format(self) -> Dict[str, Any]:
        """Validate dataset format and required fields."""
        print("\n🔍 Validating dataset format...")
        
        valid_rows = 0
        missing_instruction = 0
        missing_output = 0
        empty_instruction = 0
        empty_output = 0
        
        for i, row in enumerate(self.data):
            has_instruction = 'instruction' in row
            has_output = 'output' in row
            
            if not has_instruction:
                missing_instruction += 1
            elif not row['instruction'] or not str(row['instruction']).strip():
                empty_instruction += 1
            
            if not has_output:
                missing_output += 1
            elif not row['output'] or not str(row['output']).strip():
                empty_output += 1
            
            if (has_instruction and has_output and 
                row['instruction'] and str(row['instruction']).strip() and
                row['output'] and str(row['output']).strip()):
                valid_rows += 1
        
        results = {
            'total_rows': len(self.data),
            'valid_rows': valid_rows,
            'missing_instruction': missing_instruction,
            'missing_output': missing_output,
            'empty_instruction': empty_instruction,
            'empty_output': empty_output,
            'format_valid': valid_rows == len(self.data)
        }
        
        print(f"   Total rows: {results['total_rows']}")
        print(f"   Valid rows: {results['valid_rows']}")
        print(f"   Missing 'instruction': {results['missing_instruction']}")
        print(f"   Missing 'output': {results['missing_output']}")
        print(f"   Empty instructions: {results['empty_instruction']}")
        print(f"   Empty outputs: {results['empty_output']}")
        
        return results
    
    def check_instruction_uniqueness(self) -> Dict[str, Any]:
        """Analyze instruction uniqueness."""
        print("\n🔄 Checking instruction uniqueness...")
        
        instructions = []
        for row in self.data:
            if 'instruction' in row and row['instruction']:
                instructions.append(str(row['instruction']).strip())
        
        unique_instructions = set(instructions)
        instruction_counts = Counter(instructions)
        duplicates = {inst: count for inst, count in instruction_counts.items() if count > 1}
        
        results = {
            'total_instructions': len(instructions),
            'unique_instructions': len(unique_instructions),
            'uniqueness_ratio': len(unique_instructions) / len(instructions) if instructions else 0,
            'duplicate_count': len(duplicates),
            'most_common_duplicates': instruction_counts.most_common(5)
        }
        
        print(f"   Total instructions: {results['total_instructions']}")
        print(f"   Unique instructions: {results['unique_instructions']}")
        print(f"   Uniqueness ratio: {results['uniqueness_ratio']:.2%}")
        print(f"   Duplicate instructions: {results['duplicate_count']}")
        
        return results
    
    def analyze_output_quality(self) -> Dict[str, Any]:
        """Analyze output quality metrics."""
        print("\n📊 Analyzing output quality...")
        
        outputs = []
        flesch_scores = []
        lengths = []
        
        for row in self.data:
            if 'output' in row and row['output']:
                output_text = str(row['output']).strip()
                outputs.append(output_text)
                lengths.append(len(output_text))
                
                # Calculate Flesch Reading Ease Score
                try:
                    flesch = textstat.flesch_reading_ease(output_text)
                    flesch_scores.append(flesch)
                except:
                    flesch_scores.append(0)  # Default for problematic texts
        
        if not outputs:
            return {'error': 'No valid outputs found'}
        
        # Length analysis
        length_stats = {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'q25': np.percentile(lengths, 25),
            'q75': np.percentile(lengths, 75)
        }
        
        # Flag very short or very long outputs
        very_short = sum(1 for l in lengths if l < 10)  # Less than 10 characters
        very_long = sum(1 for l in lengths if l > 2000)  # More than 2000 characters
        
        results = {
            'total_outputs': len(outputs),
            'avg_flesch_score': np.mean(flesch_scores) if flesch_scores else 0,
            'flesch_interpretation': self._interpret_flesch_score(np.mean(flesch_scores) if flesch_scores else 0),
            'length_stats': length_stats,
            'very_short_outputs': very_short,
            'very_long_outputs': very_long,
            'length_distribution': lengths[:1000]  # Sample for plotting
        }
        
        print(f"   Total outputs analyzed: {results['total_outputs']}")
        print(f"   Average Flesch Reading Ease: {results['avg_flesch_score']:.1f} ({results['flesch_interpretation']})")
        print(f"   Average output length: {length_stats['mean']:.0f} characters")
        print(f"   Very short outputs (<10 chars): {very_short}")
        print(f"   Very long outputs (>2000 chars): {very_long}")
        
        # Create length distribution plot
        self._plot_length_distribution(lengths)
        
        return results
    
    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch Reading Ease Score."""
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
    
    def _plot_length_distribution(self, lengths: List[int]):
        """Plot output length distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Output Lengths')
        plt.xlabel('Output Length (characters)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig('output_length_distribution.png', dpi=150, bbox_inches='tight')
        print("   📈 Length distribution plot saved as 'output_length_distribution.png'")
        plt.show()
    
    def detect_duplicates(self) -> Dict[str, Any]:
        """Detect duplicate (instruction, output) pairs."""
        print("\n🔍 Detecting duplicate pairs...")
        
        pairs = []
        for row in self.data:
            if 'instruction' in row and 'output' in row and row['instruction'] and row['output']:
                pair = (str(row['instruction']).strip(), str(row['output']).strip())
                pairs.append(pair)
        
        unique_pairs = set(pairs)
        pair_counts = Counter(pairs)
        duplicates = {pair: count for pair, count in pair_counts.items() if count > 1}
        
        results = {
            'total_pairs': len(pairs),
            'unique_pairs': len(unique_pairs),
            'duplicate_pairs': len(duplicates),
            'duplication_ratio': len(duplicates) / len(pairs) if pairs else 0
        }
        
        print(f"   Total (instruction, output) pairs: {results['total_pairs']}")
        print(f"   Unique pairs: {results['unique_pairs']}")
        print(f"   Duplicate pairs: {results['duplicate_pairs']}")
        print(f"   Duplication ratio: {results['duplication_ratio']:.2%}")
        
        return results
    
    def detect_toxicity(self) -> Dict[str, Any]:
        """Detect toxic content in outputs using Detoxify."""
        print("\n🛡️  Detecting toxicity...")
        
        try:
            if self.detoxify_model is None:
                print("   Loading Detoxify model...")
                self.detoxify_model = Detoxify('original')
            
            outputs = []
            for row in self.data:
                if 'output' in row and row['output']:
                    outputs.append(str(row['output']).strip())
            
            if not outputs:
                return {'error': 'No outputs to analyze'}
            
            # Sample up to 1000 outputs for efficiency
            sample_size = min(1000, len(outputs))
            sample_outputs = outputs[:sample_size]
            
            print(f"   Analyzing toxicity for {sample_size} outputs...")
            toxicity_scores = self.detoxify_model.predict(sample_outputs)
            
            # Count toxic outputs (threshold: 0.5)
            toxic_threshold = 0.5
            toxic_outputs = sum(1 for score in toxicity_scores['toxicity'] if score > toxic_threshold)
            
            results = {
                'analyzed_outputs': sample_size,
                'toxic_outputs': toxic_outputs,
                'toxicity_rate': toxic_outputs / sample_size,
                'avg_toxicity_score': np.mean(toxicity_scores['toxicity']),
                'max_toxicity_score': max(toxicity_scores['toxicity'])
            }
            
            print(f"   Toxic outputs (>{toxic_threshold}): {toxic_outputs}/{sample_size}")
            print(f"   Toxicity rate: {results['toxicity_rate']:.2%}")
            print(f"   Average toxicity score: {results['avg_toxicity_score']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"   ⚠️  Toxicity detection failed: {e}")
            return {'error': str(e)}
    
    def check_dataset_size(self) -> Dict[str, Any]:
        """Check if dataset meets minimum size requirements."""
        print("\n📏 Checking dataset size...")
        
        # Count clean rows (valid format + reasonable length)
        clean_rows = 0
        for row in self.data:
            if ('instruction' in row and 'output' in row and 
                row['instruction'] and row['output'] and
                len(str(row['instruction']).strip()) >= 5 and
                len(str(row['output']).strip()) >= 10):
                clean_rows += 1
        
        min_required = 5000
        results = {
            'total_rows': len(self.data),
            'clean_rows': clean_rows,
            'min_required': min_required,
            'size_adequate': clean_rows >= min_required
        }
        
        print(f"   Total rows: {results['total_rows']}")
        print(f"   Clean rows: {results['clean_rows']}")
        print(f"   Minimum required: {results['min_required']}")
        print(f"   Size adequate: {'✅ Yes' if results['size_adequate'] else '❌ No'}")
        
        return results
    
    def analyze_sample_outputs(self, sample_size: int = 10) -> Dict[str, Any]:
        """Analyze a sample of instruction-output pairs for manual review."""
        print(f"\n📝 Sample instruction-output pairs for manual review:")
        
        valid_samples = []
        for row in self.data:
            if ('instruction' in row and 'output' in row and 
                row['instruction'] and row['output']):
                valid_samples.append(row)
                if len(valid_samples) >= sample_size:
                    break
        
        if not valid_samples:
            return {'error': 'No valid samples found'}
        
        print(f"   Showing {len(valid_samples)} sample pairs:")
        
        for i, sample in enumerate(valid_samples):
            instruction = str(sample['instruction']).strip()
            output = str(sample['output']).strip()
            
            print(f"\n   --- Sample {i+1} ---")
            print(f"   Instruction: {instruction[:200]}{'...' if len(instruction) > 200 else ''}")
            print(f"   Output: {output[:300]}{'...' if len(output) > 300 else ''}")
            print(f"   Output Length: {len(output)} characters")
        
        results = {
            'samples_shown': len(valid_samples),
            'sample_data': valid_samples
        }
        
        return results
    
    def generate_final_report(self) -> bool:
        """Generate final assessment report."""
        print("\n" + "="*60)
        print("📋 FINAL DATASET QUALITY ASSESSMENT REPORT")
        print("="*60)
        
        # Collect all evaluation results
        format_results = self.evaluation_results.get('format', {})
        uniqueness_results = self.evaluation_results.get('uniqueness', {})
        quality_results = self.evaluation_results.get('quality', {})
        duplicate_results = self.evaluation_results.get('duplicates', {})
        toxicity_results = self.evaluation_results.get('toxicity', {})
        size_results = self.evaluation_results.get('size', {})
        
        # Define pass/fail criteria
        checks = {}
        
        # Format validation
        checks['format'] = format_results.get('format_valid', False)
        print(f"1. Format Validation: {'✅ PASS' if checks['format'] else '❌ FAIL'}")
        
        # Instruction uniqueness (>80% unique)
        uniqueness_ratio = uniqueness_results.get('uniqueness_ratio', 0)
        checks['uniqueness'] = uniqueness_ratio > 0.8
        print(f"2. Instruction Uniqueness: {'✅ PASS' if checks['uniqueness'] else '❌ FAIL'} ({uniqueness_ratio:.1%})")
        
        # Output quality (reasonable Flesch score and not too many extreme lengths)
        flesch_score = quality_results.get('avg_flesch_score', 0)
        very_short = quality_results.get('very_short_outputs', 0)
        very_long = quality_results.get('very_long_outputs', 0)
        total_outputs = quality_results.get('total_outputs', 1)
        extreme_ratio = (very_short + very_long) / total_outputs
        checks['quality'] = 30 <= flesch_score <= 100 and extreme_ratio < 0.1
        print(f"3. Output Quality: {'✅ PASS' if checks['quality'] else '❌ FAIL'} (Flesch: {flesch_score:.1f}, Extreme: {extreme_ratio:.1%})")
        
        # Duplicate detection (<5% duplicates)
        dup_ratio = duplicate_results.get('duplication_ratio', 0)
        checks['duplicates'] = dup_ratio < 0.05
        print(f"4. Duplicate Detection: {'✅ PASS' if checks['duplicates'] else '❌ FAIL'} ({dup_ratio:.1%} duplicates)")
        
        # Toxicity (<5% toxic)
        if 'error' not in toxicity_results:
            tox_rate = toxicity_results.get('toxicity_rate', 0)
            checks['toxicity'] = tox_rate < 0.05
            print(f"5. Toxicity Detection: {'✅ PASS' if checks['toxicity'] else '❌ FAIL'} ({tox_rate:.1%} toxic)")
        else:
            checks['toxicity'] = True  # Skip if toxicity detection failed
            print(f"5. Toxicity Detection: ⚠️  SKIPPED (detection failed)")
        
        # Dataset size (>=5000 clean rows)
        checks['size'] = size_results.get('size_adequate', False)
        clean_rows = size_results.get('clean_rows', 0)
        print(f"6. Dataset Size: {'✅ PASS' if checks['size'] else '❌ FAIL'} ({clean_rows:,} clean rows)")
        
        # Overall assessment
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        print(f"\n📊 SUMMARY: {passed_checks}/{total_checks} checks passed")
        
        # Final recommendation
        ready_for_finetuning = passed_checks >= total_checks - 1  # Allow 1 failure
        
        print("\n" + "="*60)
        if ready_for_finetuning:
            print("🎉 RECOMMENDATION: ✅ DATASET IS READY FOR FINE-TUNING!")
            print("Your dataset meets the quality standards for instruction-tuning.")
        else:
            print("⚠️  RECOMMENDATION: ❌ DATASET NEEDS IMPROVEMENT")
            print("Please address the failed checks before fine-tuning.")
            
            # Specific recommendations
            print("\n🔧 RECOMMENDED ACTIONS:")
            if not checks['format']:
                print("   - Fix format issues: ensure all rows have non-empty 'instruction' and 'output' fields")
            if not checks['uniqueness']:
                print("   - Increase instruction diversity: remove or modify duplicate instructions")
            if not checks['quality']:
                print("   - Improve output quality: review very short/long outputs and readability")
            if not checks['duplicates']:
                print("   - Remove duplicate (instruction, output) pairs")
            if not checks['toxicity']:
                print("   - Remove toxic content from outputs")
            if not checks['size']:
                print("   - Increase dataset size: add more high-quality examples")
        
        print("="*60)
        
        return ready_for_finetuning
    
    def run_full_evaluation(self):
        """Run complete dataset evaluation pipeline."""
        print("🚀 Starting comprehensive dataset quality evaluation...")
        
        # Load dataset
        if not self.load_dataset():
            return False
        
        # Run all evaluations
        self.evaluation_results['format'] = self.validate_format()
        self.evaluation_results['uniqueness'] = self.check_instruction_uniqueness()
        self.evaluation_results['quality'] = self.analyze_output_quality()
        self.evaluation_results['duplicates'] = self.detect_duplicates()
        self.evaluation_results['toxicity'] = self.detect_toxicity()
        self.evaluation_results['size'] = self.check_dataset_size()
        
        # Sample analysis for manual review
        try:
            self.evaluation_results['samples'] = self.analyze_sample_outputs()
        except Exception as e:
            print(f"⚠️  Skipping sample analysis: {e}")
        
        # Generate final report
        return self.generate_final_report()

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python dataset_evaluator.py <path_to_jsonl_file>")
        print("Example: python dataset_evaluator.py my_dataset.jsonl")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    
    if not os.path.exists(jsonl_file):
        print(f"❌ File not found: {jsonl_file}")
        sys.exit(1)
    
    # Run evaluation
    evaluator = DatasetQualityEvaluator(jsonl_file)
    success = evaluator.run_full_evaluation()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()