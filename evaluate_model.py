
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration - UPDATE THESE PATHS
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAINED_MODEL_PATH = "./mistral-7b-beekeeping-lora"  # Your trained model path
TEST_DATASET_PATH = "test_dataset.jsonl"  # NEW TEST DATA (create this)
HF_TOKEN = getenv("hf")

# Set token
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

def create_sample_test_data():
    """Create sample test data for beekeeping evaluation"""
    print("📝 Creating sample test data...")
    
    # Sample beekeeping questions NOT in your training data
    test_questions = [
        {
            "instruction": "What equipment is essential for a beginner beekeeper?",
            "expected_topics": ["hive tool", "smoker", "protective gear", "frames", "foundation"]
        },
        {
            "instruction": "How do you identify if your hive has been robbed by other bees?",
            "expected_topics": ["dead bees", "wax cappings", "reduced honey stores", "aggressive behavior"]
        },
        {
            "instruction": "What are the main differences between Italian and Carniolan bee breeds?",
            "expected_topics": ["temperament", "brood pattern", "honey production", "wintering ability"]
        },
        {
            "instruction": "How should you prepare your hives for winter in cold climates?",
            "expected_topics": ["insulation", "ventilation", "food stores", "entrance reducers", "windbreaks"]
        },
        {
            "instruction": "What causes chalkbrood disease and how is it treated?",
            "expected_topics": ["fungal infection", "Ascosphaera apis", "moisture control", "hive ventilation"]
        },
        {
            "instruction": "When is the best time to split a strong hive?",
            "expected_topics": ["spring", "drone cells", "queen cells", "population", "nectar flow"]
        },
        {
            "instruction": "How do you assess the quality of a queen bee?",
            "expected_topics": ["egg laying pattern", "brood pattern", "pheromones", "worker behavior"]
        },
        {
            "instruction": "What plants are best for supporting bee populations throughout the season?",
            "expected_topics": ["diverse bloom times", "native plants", "clover", "wildflowers", "fruit trees"]
        },
        {
            "instruction": "How do you safely remove bees from a structure without killing them?",
            "expected_topics": ["live removal", "bee vacuum", "relocation", "comb removal", "prevention"]
        },
        {
            "instruction": "What are the signs that indicate your hive is preparing to swarm?",
            "expected_topics": ["queen cells", "crowded conditions", "reduced foraging", "clustering"]
        }
    ]
    
    # Save as JSONL
    with open(TEST_DATASET_PATH, 'w', encoding='utf-8') as f:
        for item in test_questions:
            f.write(json.dumps(item) + '\n')
    
    print(f"✅ Created {len(test_questions)} test questions in {TEST_DATASET_PATH}")
    return test_questions

def load_test_data():
    """Load test data from JSONL file"""
    print("📁 Loading test data...")
    
    if not os.path.exists(TEST_DATASET_PATH):
        print(f"❌ Test dataset not found: {TEST_DATASET_PATH}")
        print("Creating sample test data...")
        return create_sample_test_data()
    
    data = []
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"✅ Loaded {len(data)} test samples")
    return data

def setup_quantization():
    """Setup quantization config"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_models():
    """Load both base model and fine-tuned model for comparison"""
    print("🤖 Loading models...")
    
    # Check if fine-tuned model exists
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"❌ Trained model not found: {TRAINED_MODEL_PATH}")
        print("Make sure your model was saved correctly after training.")
        exit(1)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model (for comparison)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=setup_quantization(),
        device_map="auto",
        token=HF_TOKEN
    )
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=setup_quantization(),
        device_map="auto",
        token=HF_TOKEN
    )
    ft_model = PeftModel.from_pretrained(ft_base_model, TRAINED_MODEL_PATH)
    
    print("✅ Models loaded successfully!")
    return base_model, ft_model, tokenizer

def generate_response(model, tokenizer, instruction, max_tokens=150):
    """Generate response for a given instruction"""
    prompt = f"<s>[INST] {instruction} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("[/INST]")[-1].strip()

def evaluate_relevance(response, expected_topics):
    """Simple relevance evaluation based on topic coverage"""
    response_lower = response.lower()
    covered_topics = 0
    
    for topic in expected_topics:
        if topic.lower() in response_lower:
            covered_topics += 1
    
    relevance_score = covered_topics / len(expected_topics) if expected_topics else 0
    return relevance_score, covered_topics, len(expected_topics)

def evaluate_models(base_model, ft_model, tokenizer, test_data):
    """Evaluate both models and compare performance"""
    print(f"🔍 Evaluating models on {len(test_data)} test samples...")
    
    results = {
        'base_model': {'responses': [], 'relevance_scores': [], 'avg_length': []},
        'fine_tuned': {'responses': [], 'relevance_scores': [], 'avg_length': []}
    }
    
    detailed_results = []
    
    for i, item in enumerate(test_data):
        print(f"  Processing sample {i+1}/{len(test_data)}...")
        
        instruction = item['instruction']
        expected_topics = item.get('expected_topics', [])
        
        # Generate responses from both models
        base_response = generate_response(base_model, tokenizer, instruction)
        ft_response = generate_response(ft_model, tokenizer, instruction)
        
        # Evaluate relevance
        base_relevance, base_covered, total_topics = evaluate_relevance(base_response, expected_topics)
        ft_relevance, ft_covered, _ = evaluate_relevance(ft_response, expected_topics)
        
        # Store results
        results['base_model']['responses'].append(base_response)
        results['base_model']['relevance_scores'].append(base_relevance)
        results['base_model']['avg_length'].append(len(base_response.split()))
        
        results['fine_tuned']['responses'].append(ft_response)
        results['fine_tuned']['relevance_scores'].append(ft_relevance)
        results['fine_tuned']['avg_length'].append(len(ft_response.split()))
        
        # Detailed comparison
        detailed_results.append({
            'question': instruction,
            'expected_topics': expected_topics,
            'base_model': {
                'response': base_response,
                'relevance_score': base_relevance,
                'topics_covered': f"{base_covered}/{total_topics}",
                'length': len(base_response.split())
            },
            'fine_tuned': {
                'response': ft_response,
                'relevance_score': ft_relevance,
                'topics_covered': f"{ft_covered}/{total_topics}",
                'length': len(ft_response.split())
            }
        })
    
    return results, detailed_results

def calculate_metrics(results):
    """Calculate summary metrics"""
    metrics = {}
    
    for model_name in ['base_model', 'fine_tuned']:
        model_results = results[model_name]
        
        metrics[model_name] = {
            'avg_relevance_score': np.mean(model_results['relevance_scores']),
            'avg_response_length': np.mean(model_results['avg_length']),
            'relevance_std': np.std(model_results['relevance_scores']),
            'num_samples': len(model_results['responses'])
        }
    
    # Calculate improvement
    improvement = {
        'relevance_improvement': metrics['fine_tuned']['avg_relevance_score'] - metrics['base_model']['avg_relevance_score'],
        'length_change': metrics['fine_tuned']['avg_response_length'] - metrics['base_model']['avg_response_length']
    }
    
    return metrics, improvement

def save_results(metrics, improvement, detailed_results):
    """Save evaluation results"""
    print("💾 Saving results...")
    
    # Create results directory
    results_dir = os.path.join(TRAINED_MODEL_PATH, "test_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics summary
    summary = {
        'metrics': metrics,
        'improvement': improvement,
        'evaluation_type': 'unseen_test_data',
        'note': 'This evaluation uses NEW test data not seen during training'
    }
    
    with open(os.path.join(results_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(results_dir, "detailed_comparison.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"✅ Results saved to {results_dir}/")
    return results_dir

def display_results(metrics, improvement, detailed_results):
    """Display evaluation results"""
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS (Using Unseen Test Data)")
    print("="*80)
    
    print(f"\n🤖 BASE MODEL PERFORMANCE:")
    print(f"  Average Relevance Score: {metrics['base_model']['avg_relevance_score']:.3f}")
    print(f"  Average Response Length: {metrics['base_model']['avg_response_length']:.1f} words")
    print(f"  Relevance Std Dev: {metrics['base_model']['relevance_std']:.3f}")
    
    print(f"\n🎯 FINE-TUNED MODEL PERFORMANCE:")
    print(f"  Average Relevance Score: {metrics['fine_tuned']['avg_relevance_score']:.3f}")
    print(f"  Average Response Length: {metrics['fine_tuned']['avg_response_length']:.1f} words")
    print(f"  Relevance Std Dev: {metrics['fine_tuned']['relevance_std']:.3f}")
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Relevance Improvement: {improvement['relevance_improvement']:+.3f}")
    print(f"  Length Change: {improvement['length_change']:+.1f} words")
    
    # Show percentage improvement
    if metrics['base_model']['avg_relevance_score'] > 0:
        pct_improvement = (improvement['relevance_improvement'] / metrics['base_model']['avg_relevance_score']) * 100
        print(f"  Relative Improvement: {pct_improvement:+.1f}%")
    
    print(f"\n📝 SAMPLE COMPARISONS:")
    print("-" * 80)
    
    # Show first 3 detailed comparisons
    for i, result in enumerate(detailed_results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {result['question']}")
        print(f"Expected Topics: {', '.join(result['expected_topics'])}")
        print(f"\nBase Model ({result['base_model']['topics_covered']} topics):")
        print(f"  {result['base_model']['response'][:200]}...")
        print(f"\nFine-tuned ({result['fine_tuned']['topics_covered']} topics):")
        print(f"  {result['fine_tuned']['response'][:200]}...")

def main():
    """Main evaluation function"""
    print("🔍 PROPER MODEL EVALUATION (Using Unseen Test Data)")
    print("This evaluation prevents overfitting by using NEW test questions")
    print("-" * 80)
    
    # Load test data
    test_data = load_test_data()
    
    # Load models
    base_model, ft_model, tokenizer = load_models()
    
    # Evaluate models
    results, detailed_results = evaluate_models(base_model, ft_model, tokenizer, test_data)
    
    # Calculate metrics
    metrics, improvement = calculate_metrics(results)
    
    # Save results
    save_results(metrics, improvement, detailed_results)
    
    # Display results
    display_results(metrics, improvement, detailed_results)
    
    print(f"\n✅ Evaluation completed! Check the results in {TRAINED_MODEL_PATH}/test_evaluation/")

if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GPU recommended for evaluation.")
    else:
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    
    main()