import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

# Set your Hugging Face token
HF = getenv("hf")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF

# Configuration - Update these paths according to your setup
class EvalConfig:
    def __init__(self):
        # Model settings
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.fine_tuned_model_path = "./mistral-7b-beekeeping-lora"  # Path to your fine-tuned model
        self.dataset_path = "training_dataset.jsonl"  # Path to your original dataset
        self.test_dataset_path = None  # Set this if you have a separate test file
        
        # Evaluation settings
        self.max_samples = 100  # Number of samples to evaluate (set to None for all)
        self.max_new_tokens = 150
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Output settings
        self.output_dir = "./evaluation_results"

# Create config instance
config = EvalConfig()

def setup_quantization_config():
    """Setup quantization config for loading the fine-tuned model"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_test_data():
    """Load test data from the dataset"""
    print("Loading test dataset...")
    
    # Check if separate test file exists
    if config.test_dataset_path and os.path.exists(config.test_dataset_path):
        print(f"Using separate test file: {config.test_dataset_path}")
        with open(config.test_dataset_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line.strip()) for line in f if line.strip()]
    else:
        # Load from main dataset and use the same split logic as training
        print(f"Using test split from: {config.dataset_path}")
        if not os.path.exists(config.dataset_path):
            print(f"❌ Dataset file not found: {config.dataset_path}")
            exit(1)
        
        with open(config.dataset_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line.strip()) for line in f if line.strip()]
        
        # Use the same 90/10 split as in training (last 10% as test)
        test_size = int(len(all_data) * 0.1)
        test_data = all_data[-test_size:]
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Limit samples if specified
    if config.max_samples and len(test_data) > config.max_samples:
        test_data = test_data[:config.max_samples]
        print(f"Limited to {len(test_data)} samples for evaluation")
    
    return test_data

def load_models():
    """Load both base model and fine-tuned model"""
    print("Loading models...")
    
    # Set HF token
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print("❌ HUGGINGFACE_HUB_TOKEN environment variable required!")
        exit(1)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=hf_token,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Skip base model loading to avoid memory issues - just focus on fine-tuned model
    print("Skipping base model loading to avoid device conflicts...")
    base_model = None
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    if not os.path.exists(config.fine_tuned_model_path):
        print(f"❌ Fine-tuned model not found: {config.fine_tuned_model_path}")
        exit(1)
    
    # Load base model with quantization for fine-tuned version
    ft_base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=setup_quantization_config(),
        device_map="auto",
        token=hf_token,
        trust_remote_code=True
    )
    
    # Load LoRA weights
    ft_model = PeftModel.from_pretrained(ft_base_model, config.fine_tuned_model_path)
    
    # Ensure model is in eval mode
    ft_model.eval()
    
    print("✅ Fine-tuned model loaded successfully")
    return tokenizer, base_model, ft_model

def generate_response(model, tokenizer, prompt, device=None):
    """Generate response from model"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model device (first parameter's device)
    model_device = next(model.parameters()).device
    
    # Move inputs to model device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=True,
            top_p=config.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after [/INST])
    if "[/INST]" in full_response:
        response = full_response.split("[/INST]")[-1].strip()
    else:
        response = full_response.strip()
    
    return response

def calculate_metrics(predictions, references):
    """Calculate BLEU and ROUGE scores"""
    print("Calculating metrics...")
    
    # Initialize scorers
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4
    
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        # BLEU score
        try:
            bleu = sentence_bleu(
                [ref.split()], 
                pred.split(),
                smoothing_function=smoothie
            )
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)
        
        # ROUGE scores
        try:
            rouge_result = rouge_scorer_obj.score(ref, pred)
            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
        except:
            rouge_scores['rouge1'].append(0.0)
            rouge_scores['rouge2'].append(0.0)
            rouge_scores['rougeL'].append(0.0)
    
    return {
        'bleu_score': np.mean(bleu_scores),
        'rouge1_score': np.mean(rouge_scores['rouge1']),
        'rouge2_score': np.mean(rouge_scores['rouge2']),
        'rougeL_score': np.mean(rouge_scores['rougeL']),
        'individual_bleu': bleu_scores,
        'individual_rouge': rouge_scores
    }

def evaluate_fine_tuned_model(tokenizer, ft_model, test_data):
    """Evaluate the fine-tuned model"""
    print(f"\n🔍 Evaluating fine-tuned model on {len(test_data)} samples...")
    
    predictions = []
    references = []
    
    ft_model.eval()
    
    for i, item in enumerate(test_data):
        if i % 20 == 0:
            print(f"Processed {i}/{len(test_data)} samples...")
        
        # Create prompt
        prompt = f"<s>[INST] {item['instruction']} [/INST]"
        
        # Generate response
        response = generate_response(ft_model, tokenizer, prompt)
        
        predictions.append(response)
        references.append(item['output'].strip())
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    
    print("\n📊 Fine-tuned Model Results:")
    print(f"BLEU Score: {metrics['bleu_score']:.4f}")
    print(f"ROUGE-1 Score: {metrics['rouge1_score']:.4f}")
    print(f"ROUGE-2 Score: {metrics['rouge2_score']:.4f}")
    print(f"ROUGE-L Score: {metrics['rougeL_score']:.4f}")
    
    return predictions, references, metrics

def compare_models(tokenizer, base_model, ft_model, test_data, num_samples=10):
    """Compare base model vs fine-tuned model"""
    if base_model is None:
        print("⚠️ Base model not available for comparison")
        return None
    
    print(f"\n🆚 Comparing models on {num_samples} samples...")
    
    # Limit samples for comparison
    comparison_data = test_data[:num_samples]
    
    base_predictions = []
    ft_predictions = []
    references = []
    
    base_model.eval()
    ft_model.eval()
    
    for i, item in enumerate(comparison_data):
        print(f"Comparing sample {i+1}/{len(comparison_data)}...")
        
        prompt = f"<s>[INST] {item['instruction']} [/INST]"
        
        # Base model response
        base_response = generate_response(base_model, tokenizer, prompt)
        base_predictions.append(base_response)
        
        # Fine-tuned model response
        ft_response = generate_response(ft_model, tokenizer, prompt)
        ft_predictions.append(ft_response)
        
        references.append(item['output'].strip())
    
    # Calculate metrics for both models
    base_metrics = calculate_metrics(base_predictions, references)
    ft_metrics = calculate_metrics(ft_predictions, references)
    
    print("\n📊 Model Comparison Results:")
    print(f"Base Model BLEU: {base_metrics['bleu_score']:.4f}")
    print(f"Fine-tuned BLEU: {ft_metrics['bleu_score']:.4f}")
    print(f"Base Model ROUGE-L: {base_metrics['rougeL_score']:.4f}")
    print(f"Fine-tuned ROUGE-L: {ft_metrics['rougeL_score']:.4f}")
    
    # Create comparison results
    comparison_results = []
    for i in range(len(comparison_data)):
        comparison_results.append({
            'instruction': comparison_data[i]['instruction'],
            'reference': references[i],
            'base_model_response': base_predictions[i],
            'fine_tuned_response': ft_predictions[i],
            'base_bleu': base_metrics['individual_bleu'][i],
            'ft_bleu': ft_metrics['individual_bleu'][i]
        })
    
    return {
        'base_metrics': base_metrics,
        'ft_metrics': ft_metrics,
        'comparisons': comparison_results
    }

def main():
    """Main evaluation function"""
    print("🧪 Starting Model Evaluation")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load test data
    test_data = load_test_data()
    
    # Load models
    tokenizer, base_model, ft_model = load_models()
    
    # Evaluate fine-tuned model
    ft_predictions, references, ft_metrics = evaluate_fine_tuned_model(tokenizer, ft_model, test_data)
    
    # Compare with base model (limited samples) - only if base model is available
    comparison_results = None
    if base_model is not None:
        comparison_results = compare_models(tokenizer, base_model, ft_model, test_data, num_samples=10)
    else:
        print("⚠️ Skipping model comparison (base model not loaded)")
    
    # Save detailed results
    detailed_results = {
        'config': {
            'model_name': config.model_name,
            'fine_tuned_model_path': config.fine_tuned_model_path,
            'num_test_samples': len(test_data),
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature
        },
        'fine_tuned_metrics': ft_metrics,
        'sample_predictions': [
            {
                'instruction': test_data[i]['instruction'],
                'reference': references[i],
                'prediction': ft_predictions[i],
                'bleu': ft_metrics['individual_bleu'][i],
                'rouge1': ft_metrics['individual_rouge']['rouge1'][i],
                'rouge2': ft_metrics['individual_rouge']['rouge2'][i],
                'rougeL': ft_metrics['individual_rouge']['rougeL'][i]
            }
            for i in range(min(20, len(test_data)))  # Save first 20 examples
        ]
    }
    
    # Add comparison results if available
    if comparison_results:
        detailed_results['comparison'] = comparison_results
    
    # Save results
    results_file = os.path.join(config.output_dir, "evaluation_results.json")
    with open(results_file, "w", encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {results_file}")
    print(f"📊 Final BLEU Score: {ft_metrics['bleu_score']:.4f}")
    print(f"📊 Final ROUGE-L Score: {ft_metrics['rougeL_score']:.4f}")
    
    # Print some example predictions
    print("\n🔍 Sample Predictions:")
    for i in range(min(3, len(test_data))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {test_data[i]['instruction']}")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {ft_predictions[i]}")
        print(f"BLEU: {ft_metrics['individual_bleu'][i]:.3f}")

if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ No GPU available, using CPU (will be slow)")
    
    # Run evaluation
    main()