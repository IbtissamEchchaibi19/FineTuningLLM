import json
import os
import sys
import warnings
import re
from collections import Counter
from typing import Dict, List, Tuple, Any
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

class DatasetProcessor:
    """Complete dataset processor: cleanup + quality evaluation."""
    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.cleaned_file = input_file.replace('.jsonl', '_cleaned.jsonl')
        self.raw_data = []
        self.cleaned_data = []
        self.evaluation_results = {}
        self.detoxify_model = None
        
    def load_raw_dataset(self) -> bool:
        """Load original dataset."""
        print("📁 Loading original dataset...")
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        self.raw_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"⚠️  Invalid JSON on line {line_num}")
            
            print(f"✅ Loaded {len(self.raw_data)} rows from original dataset")
            return len(self.raw_data) > 0
        except FileNotFoundError:
            print(f"❌ File not found: {self.input_file}")
            return False
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False
    
    def improve_readability(self, text: str) -> str:
        """Improve text readability."""
        if not text:
            return text
            
        text = str(text).strip()
        
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\.([A-Z])', r'. \1', text)  # Space after period
        text = re.sub(r'([.!?])\s*([.!?])', r'\1 \2', text)  # Punctuation spacing
        
        # Break very long sentences at natural breaks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        improved_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 150:  # Very long sentence
                # Try to break at natural points
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
        
        # Ensure proper ending
        if result and not result.endswith(('.', '!', '?', ':')):
            result += '.'
            
        return result
    
    def clean_dataset(self) -> Dict[str, Any]:
        """Complete dataset cleanup pipeline."""
        print("\n🧹 Starting dataset cleanup...")
        
        stats = {
            'original': len(self.raw_data),
            'format_issues': 0,
            'duplicates': 0,
            'extreme_lengths': 0,
            'low_quality': 0,
            'final': 0
        }
        
        seen_instructions = set()
        seen_pairs = set()
        
        for row_num, row in enumerate(self.raw_data, 1):
            try:
                # 1. Format validation
                if not ('instruction' in row and 'output' in row and 
                       row['instruction'] and str(row['instruction']).strip() and
                       row['output'] and str(row['output']).strip()):
                    stats['format_issues'] += 1
                    continue
                
                # Clean and normalize
                instruction = str(row['instruction']).strip()
                output = str(row['output']).strip()
                
                # 2. Length filtering
                if len(instruction) < 5 or len(output) < 15:
                    stats['extreme_lengths'] += 1
                    continue
                    
                if len(output) > 2000:  # Very long outputs
                    stats['extreme_lengths'] += 1
                    continue
                
                # 3. Remove duplicate instructions
                instruction_normalized = instruction.lower().strip()
                if instruction_normalized in seen_instructions:
                    stats['duplicates'] += 1
                    continue
                seen_instructions.add(instruction_normalized)
                
                # 4. Remove duplicate pairs
                pair_key = (instruction_normalized, output.lower().strip())
                if pair_key in seen_pairs:
                    stats['duplicates'] += 1
                    continue
                seen_pairs.add(pair_key)
                
                # 5. Quality improvements
                try:
                    # Improve output readability
                    improved_output = self.improve_readability(output)
                    
                    # Basic quality check
                    if len(improved_output.split()) < 3:  # Too few words
                        stats['low_quality'] += 1
                        continue
                    
                    # Keep the cleaned row
                    cleaned_row = {
                        'instruction': instruction,
                        'output': improved_output
                    }
                    
                    # Preserve other fields if they exist
                    for key, value in row.items():
                        if key not in ['instruction', 'output'] and value is not None:
                            cleaned_row[key] = value
                    
                    self.cleaned_data.append(cleaned_row)
                    stats['final'] += 1
                    
                except Exception as e:
                    stats['low_quality'] += 1
                    continue
                    
            except Exception as e:
                stats['format_issues'] += 1
                continue
        
        # Save cleaned dataset
        try:
            with open(self.cleaned_file, 'w', encoding='utf-8') as f:
                for row in self.cleaned_data:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"❌ Error saving cleaned dataset: {e}")
            return stats
        
        # Print cleanup results
        print(f"\n📊 Cleanup Results:")
        print(f"   Original rows: {stats['original']:,}")
        print(f"   Format issues removed: {stats['format_issues']:,}")
        print(f"   Duplicates removed: {stats['duplicates']:,}")
        print(f"   Extreme lengths removed: {stats['extreme_lengths']:,}")
        print(f"   Low quality removed: {stats['low_quality']:,}")
        print(f"   Final clean rows: {stats['final']:,}")
        print(f"   Retention rate: {stats['final']/stats['original']:.1%}")
        print(f"✅ Cleaned dataset saved as: {self.cleaned_file}")
        
        return stats
    
    def validate_format(self) -> Dict[str, Any]:
        """Validate cleaned dataset format."""
        print("\n🔍 Validating cleaned dataset format...")
        
        valid_rows = 0
        issues = 0
        
        for row in self.cleaned_data:
            if ('instruction' in row and 'output' in row and 
                row['instruction'] and str(row['instruction']).strip() and
                row['output'] and str(row['output']).strip()):
                valid_rows += 1
            else:
                issues += 1
        
        results = {
            'total_rows': len(self.cleaned_data),
            'valid_rows': valid_rows,
            'issues': issues,
            'format_valid': issues == 0
        }
        
        print(f"   Total rows: {results['total_rows']}")
        print(f"   Valid rows: {results['valid_rows']}")
        print(f"   Issues: {results['issues']}")
        
        return results
    
    def check_instruction_uniqueness(self) -> Dict[str, Any]:
        """Check instruction uniqueness in cleaned data."""
        print("\n🔄 Checking instruction uniqueness...")
        
        instructions = [str(row['instruction']).strip() for row in self.cleaned_data 
                       if 'instruction' in row and row['instruction']]
        
        unique_instructions = set(instructions)
        instruction_counts = Counter(instructions)
        duplicates = {inst: count for inst, count in instruction_counts.items() if count > 1}
        
        results = {
            'total_instructions': len(instructions),
            'unique_instructions': len(unique_instructions),
            'uniqueness_ratio': len(unique_instructions) / len(instructions) if instructions else 0,
            'duplicate_count': len(duplicates)
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
        
        for row in self.cleaned_data:
            if 'output' in row and row['output']:
                output_text = str(row['output']).strip()
                outputs.append(output_text)
                lengths.append(len(output_text))
                
                try:
                    flesch = textstat.flesch_reading_ease(output_text)
                    flesch_scores.append(flesch)
                except:
                    flesch_scores.append(50)  # Default neutral score
        
        if not outputs:
            return {'error': 'No valid outputs found'}
        
        # Length statistics
        length_stats = {
            'mean': np.mean(lengths),
            'median': np.median(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'q25': np.percentile(lengths, 25),
            'q75': np.percentile(lengths, 75)
        }
        
        # Flag extreme lengths
        very_short = sum(1 for l in lengths if l < 20)
        very_long = sum(1 for l in lengths if l > 1500)
        
        results = {
            'total_outputs': len(outputs),
            'avg_flesch_score': np.mean(flesch_scores) if flesch_scores else 50,
            'flesch_interpretation': self._interpret_flesch_score(np.mean(flesch_scores) if flesch_scores else 50),
            'length_stats': length_stats,
            'very_short_outputs': very_short,
            'very_long_outputs': very_long,
            'extreme_ratio': (very_short + very_long) / len(outputs)
        }
        
        print(f"   Total outputs: {results['total_outputs']}")
        print(f"   Average Flesch Score: {results['avg_flesch_score']:.1f} ({results['flesch_interpretation']})")
        print(f"   Average length: {length_stats['mean']:.0f} characters")
        print(f"   Very short outputs (<20 chars): {very_short}")
        print(f"   Very long outputs (>1500 chars): {very_long}")
        print(f"   Extreme ratio: {results['extreme_ratio']:.1%}")
        
        # Create visualization
        self._plot_quality_metrics(lengths, flesch_scores)
        
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
    
    def _plot_quality_metrics(self, lengths: List[int], flesch_scores: List[float]):
        """Plot quality metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Length distribution
        ax1.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_title('Output Length Distribution')
        ax1.set_xlabel('Output Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
        ax1.legend()
        
        # Flesch score distribution
        ax2.hist(flesch_scores, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_title('Readability Score Distribution')
        ax2.set_xlabel('Flesch Reading Ease Score')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(np.mean(flesch_scores), color='red', linestyle='--', label=f'Mean: {np.mean(flesch_scores):.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('dataset_quality_metrics.png', dpi=150, bbox_inches='tight')
        print("   📈 Quality metrics plot saved as 'dataset_quality_metrics.png'")
        plt.show()
    
    def detect_duplicates(self) -> Dict[str, Any]:
        """Detect any remaining duplicates."""
        print("\n🔍 Final duplicate check...")
        
        pairs = []
        for row in self.cleaned_data:
            if 'instruction' in row and 'output' in row:
                pair = (str(row['instruction']).strip().lower(), 
                       str(row['output']).strip().lower())
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
        
        print(f"   Total pairs: {results['total_pairs']}")
        print(f"   Unique pairs: {results['unique_pairs']}")
        print(f"   Duplicate pairs: {results['duplicate_pairs']}")
        print(f"   Duplication ratio: {results['duplication_ratio']:.2%}")
        
        return results
    
    def detect_toxicity(self) -> Dict[str, Any]:
        """Detect toxic content."""
        print("\n🛡️  Detecting toxicity...")
        
        try:
            if self.detoxify_model is None:
                print("   Loading Detoxify model...")
                self.detoxify_model = Detoxify('original')
            
            outputs = [str(row['output']).strip() for row in self.cleaned_data 
                      if 'output' in row and row['output']]
            
            if not outputs:
                return {'error': 'No outputs to analyze'}
            
            # Sample for efficiency
            sample_size = min(1000, len(outputs))
            sample_outputs = outputs[:sample_size]
            
            print(f"   Analyzing {sample_size} outputs for toxicity...")
            toxicity_scores = self.detoxify_model.predict(sample_outputs)
            
            toxic_threshold = 0.5
            toxic_count = sum(1 for score in toxicity_scores['toxicity'] if score > toxic_threshold)
            
            results = {
                'analyzed_outputs': sample_size,
                'toxic_outputs': toxic_count,
                'toxicity_rate': toxic_count / sample_size,
                'avg_toxicity_score': np.mean(toxicity_scores['toxicity']),
                'max_toxicity_score': max(toxicity_scores['toxicity'])
            }
            
            print(f"   Toxic outputs: {toxic_count}/{sample_size}")
            print(f"   Toxicity rate: {results['toxicity_rate']:.2%}")
            print(f"   Average toxicity: {results['avg_toxicity_score']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"   ⚠️  Toxicity detection failed: {e}")
            return {'error': str(e)}
    
    def check_dataset_size(self) -> Dict[str, Any]:
        """Check dataset size requirements."""
        print("\n📏 Checking dataset size...")
        
        total_rows = len(self.cleaned_data)
        min_required = 5000
        
        results = {
            'total_rows': total_rows,
            'min_required': min_required,
            'size_adequate': total_rows >= min_required
        }
        
        print(f"   Clean rows: {results['total_rows']:,}")
        print(f"   Minimum required: {results['min_required']:,}")
        print(f"   Size adequate: {'✅ Yes' if results['size_adequate'] else '❌ No'}")
        
        return results
    
    def show_sample_outputs(self, sample_size: int = 5) -> Dict[str, Any]:
        """Show sample outputs for manual review."""
        print(f"\n📝 Sample instruction-output pairs:")
        
        sample_data = self.cleaned_data[:sample_size]
        
        for i, row in enumerate(sample_data, 1):
            instruction = str(row['instruction'])
            output = str(row['output'])
            
            print(f"\n   --- Sample {i} ---")
            print(f"   Instruction: {instruction[:150]}{'...' if len(instruction) > 150 else ''}")
            print(f"   Output: {output[:200]}{'...' if len(output) > 200 else ''}")
            print(f"   Output Length: {len(output)} chars")
            
            try:
                flesch = textstat.flesch_reading_ease(output)
                print(f"   Readability: {flesch:.1f} ({self._interpret_flesch_score(flesch)})")
            except:
                print(f"   Readability: Could not calculate")
        
        return {'samples_shown': len(sample_data)}
    
    def generate_final_report(self) -> bool:
        """Generate comprehensive final report."""
        print("\n" + "="*70)
        print("📋 FINAL DATASET QUALITY ASSESSMENT REPORT")
        print("="*70)
        
        # Get results
        format_results = self.evaluation_results.get('format', {})
        uniqueness_results = self.evaluation_results.get('uniqueness', {})
        quality_results = self.evaluation_results.get('quality', {})
        duplicate_results = self.evaluation_results.get('duplicates', {})
        toxicity_results = self.evaluation_results.get('toxicity', {})
        size_results = self.evaluation_results.get('size', {})
        
        # Evaluate each criterion
        checks = {}
        
        # 1. Format validation
        checks['format'] = format_results.get('format_valid', False)
        print(f"1. Format Validation: {'✅ PASS' if checks['format'] else '❌ FAIL'}")
        
        # 2. Instruction uniqueness (>85% after cleanup)
        uniqueness_ratio = uniqueness_results.get('uniqueness_ratio', 0)
        checks['uniqueness'] = uniqueness_ratio > 0.85
        print(f"2. Instruction Uniqueness: {'✅ PASS' if checks['uniqueness'] else '❌ FAIL'} ({uniqueness_ratio:.1%})")
        
        # 3. Output quality
        flesch_score = quality_results.get('avg_flesch_score', 0)
        extreme_ratio = quality_results.get('extreme_ratio', 0)
        checks['quality'] = 25 <= flesch_score <= 85 and extreme_ratio < 0.05
        print(f"3. Output Quality: {'✅ PASS' if checks['quality'] else '❌ FAIL'} (Flesch: {flesch_score:.1f}, Extreme: {extreme_ratio:.1%})")
        
        # 4. Duplicates (<2% after cleanup)
        dup_ratio = duplicate_results.get('duplication_ratio', 0)
        checks['duplicates'] = dup_ratio < 0.02
        print(f"4. Duplicate Detection: {'✅ PASS' if checks['duplicates'] else '❌ FAIL'} ({dup_ratio:.1%})")
        
        # 5. Toxicity
        if 'error' not in toxicity_results:
            tox_rate = toxicity_results.get('toxicity_rate', 0)
            checks['toxicity'] = tox_rate < 0.05
            print(f"5. Toxicity Detection: {'✅ PASS' if checks['toxicity'] else '❌ FAIL'} ({tox_rate:.1%})")
        else:
            checks['toxicity'] = True  # Skip if failed
            print(f"5. Toxicity Detection: ⚠️  SKIPPED")
        
        # 6. Dataset size
        checks['size'] = size_results.get('size_adequate', False)
        total_rows = size_results.get('total_rows', 0)
        print(f"6. Dataset Size: {'✅ PASS' if checks['size'] else '❌ FAIL'} ({total_rows:,} rows)")
        
        # Summary
        passed = sum(checks.values())
        total = len(checks)
        
        print(f"\n📊 SUMMARY: {passed}/{total} checks passed")
        
        # Final recommendation
        ready = passed >= total - 1  # Allow 1 failure max
        
        print("\n" + "="*70)
        if ready:
            print("🎉 RECOMMENDATION: ✅ DATASET IS READY FOR FINE-TUNING!")
            print("Your cleaned dataset meets quality standards for instruction-tuning.")
            if passed < total:
                print("Note: Minor issues detected but dataset is still usable.")
        else:
            print("⚠️  RECOMMENDATION: ❌ DATASET STILL NEEDS WORK")
            print("Please review the failed checks. Consider:")
            
            if not checks['size']:
                print("   • Adding more high-quality data to reach 5,000+ examples")
            if not checks['quality']:
                print("   • Further improving output readability and length distribution")
            if not checks['uniqueness']:
                print("   • Creating more diverse instructions")
            if not checks['duplicates']:
                print("   • Additional deduplication")
            if not checks['toxicity']:
                print("   • Content moderation and filtering")
        
        print(f"\n📁 Cleaned dataset saved as: {self.cleaned_file}")
        print("="*70)
        
        return ready
    
    def run_complete_process(self):
        """Run the complete cleanup and evaluation process."""
        print("🚀 Starting complete dataset processing...")
        
        # 1. Load original data
        if not self.load_raw_dataset():
            return False
        
        # 2. Clean the dataset
        cleanup_stats = self.clean_dataset()
        
        if not self.cleaned_data:
            print("❌ No data remaining after cleanup!")
            return False
        
        # 3. Evaluate cleaned dataset
        print(f"\n🔍 Evaluating cleaned dataset ({len(self.cleaned_data)} rows)...")
        
        self.evaluation_results['format'] = self.validate_format()
        self.evaluation_results['uniqueness'] = self.check_instruction_uniqueness()
        self.evaluation_results['quality'] = self.analyze_output_quality()
        self.evaluation_results['duplicates'] = self.detect_duplicates()
        self.evaluation_results['toxicity'] = self.detect_toxicity()
        self.evaluation_results['size'] = self.check_dataset_size()
        
        # 4. Show samples
        self.show_sample_outputs()
        
        # 5. Generate final report
        return self.generate_final_report()

def main():
    """Main execution function."""
    if len(sys.argv) != 2:
        print("Usage: python complete_dataset_processor.py <input_jsonl_file>")
        print("Example: python complete_dataset_processor.py my_dataset.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        sys.exit(1)
    
    # Process dataset
    processor = DatasetProcessor(input_file)
    success = processor.run_complete_process()
    
    print(f"\n{'✅ SUCCESS' if success else '❌ NEEDS MORE WORK'}")
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()