import os
import json
import torch
import wandb
import numpy as np


HF = getenv("hf")
wB=getenv(wb)
# Set authentication tokens
os.environ["WANDB_API_KEY"] = HF
# Add your Hugging Face token here - REQUIRED for accessing gated models
os.environ["HUGGINGFACE_HUB_TOKEN"] = WB  # Replace with your actual token

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import warnings
warnings.filterwarnings("ignore")

# Configuration
class Config:
    def __init__(self):
        # Model settings - Using a more accessible model as alternative
        # Option 1: Use the original Mistral (requires HF token)
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Option 2: Use an unrestricted alternative (uncomment if you prefer)
        # self.model_name = "microsoft/DialoGPT-medium"  # Smaller, unrestricted
        # self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Also gated, but popular
        
        self.dataset_path = "training_dataset.jsonl"  # Path to your JSONL file
        self.output_dir = "./mistral-7b-beekeeping-lora"
        
        # LoRA settings
        self.lora_r = 64  # Rank - higher for better quality, lower for speed
        self.lora_alpha = 16  # Alpha scaling parameter
        self.lora_dropout = 0.1
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Using QLoRA only
        self.use_qlora = True
        
        # Training settings
        self.batch_size = 4  # Per device batch size
        self.gradient_accumulation_steps = 4  # Effective batch size = 4 * 4 = 16
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.max_seq_length = 512
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        # Logging
        self.logging_steps = 10
        self.save_steps = 100
        self.eval_steps = 100
        self.use_wandb = True  # Set to False to disable wandb logging

# Create config instance
Config = Config()

def setup_quantization_config():
    """Setup quantization config for QLoRA"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_and_prepare_dataset(tokenizer):
    """Load JSONL dataset and convert to HuggingFace format"""
    print("Loading dataset...")
    
    # Check if dataset file exists
    if not os.path.exists(Config.dataset_path):
        print(f"❌ Dataset file not found: {Config.dataset_path}")
        print("Please ensure your JSONL file exists at the specified path.")
        exit(1)
    
    # Load JSONL data
    data = []
    try:
        with open(Config.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        exit(1)
    
    print(f"Loaded {len(data)} examples")
    
    if len(data) == 0:
        print("❌ No data found in dataset file")
        exit(1)
    
    # Extract text field for training (your data already has formatted text)
    texts = [item['text'] for item in data]
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Split into train/validation (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")
    
    return dataset

def setup_model_and_tokenizer():
    """Initialize model and tokenizer with appropriate configurations"""
    print("Loading tokenizer...")
    
    # Check for HF token
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token or hf_token == "hf_your_token_here":
        print("❌ Hugging Face token not set!")
        print("Please:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a token with 'Read' permissions")
        print("3. Accept the Mistral model terms at: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2")
        print("4. Set the token in the script: os.environ['HUGGINGFACE_HUB_TOKEN'] = 'your_token'")
        exit(1)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            Config.model_name,
            token=hf_token,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("Make sure you have:")
        print("1. A valid Hugging Face token")
        print("2. Accepted the model's terms and conditions")
        print("3. Proper internet connection")
        exit(1)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model...")
    quantization_config = setup_quantization_config()
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            Config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
            # Remove flash_attention_2 as it may cause issues
            # attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("This might be due to:")
        print("1. Insufficient GPU memory")
        print("2. Model access permissions")
        print("3. Network issues")
        print("4. Outdated bitsandbytes library")
        exit(1)
    
    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def setup_lora_config():
    """Configure LoRA parameters"""
    return LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=Config.target_modules,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=Config.max_seq_length,
        return_overflowing_tokens=False,
    )

def evaluate_model_performance(model, tokenizer, test_dataset, output_dir):
    """Comprehensive evaluation of the fine-tuned model"""
    print("\n🔍 Starting Model Evaluation...")
    
    # Convert test dataset back to original format for evaluation
    eval_data = []
    with open(Config.dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    # Use last 10% as test set (same split as training)
    test_size = int(len(data) * 0.1)
    test_data = data[-test_size:]
    
    print(f"Evaluating on {len(test_data)} test samples...")
    
    # Initialize metrics
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    predictions = []
    references = []
    
    model.eval()
    
    for i, item in enumerate(test_data[:100]):  # Evaluate on first 100 samples for speed
        if i % 20 == 0:
            print(f"Processed {i}/100 samples...")
        
        # Create prompt from instruction
        prompt = f"<s>[INST] {item['instruction']} [/INST]"
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        # DON'T move inputs to device - model is already on the right device
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode prediction
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_response = full_response.split("[/INST]")[-1].strip()
        
        # Get reference
        reference_response = item['output'].strip()
        
        predictions.append(predicted_response)
        references.append(reference_response)
        
        # Calculate BLEU score
        try:
            smoothie = SmoothingFunction().method4
            bleu = sentence_bleu(
                [reference_response.split()], 
                predicted_response.split(),
                smoothing_function=smoothie
            )
            bleu_scores.append(bleu)
        except:
            bleu_scores.append(0.0)
        
        # Calculate ROUGE scores
        try:
            rouge_result = rouge_scorer_obj.score(reference_response, predicted_response)
            rouge_scores['rouge1'].append(rouge_result['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(rouge_result['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(rouge_result['rougeL'].fmeasure)
        except:
            rouge_scores['rouge1'].append(0.0)
            rouge_scores['rouge2'].append(0.0)
            rouge_scores['rougeL'].append(0.0)
    
    # Calculate final metrics
    metrics = {
        'bleu_score': np.mean(bleu_scores),
        'rouge1_score': np.mean(rouge_scores['rouge1']),
        'rouge2_score': np.mean(rouge_scores['rouge2']),
        'rougeL_score': np.mean(rouge_scores['rougeL']),
        'num_samples_evaluated': len(predictions)
    }
    
    # Print results
    print("\n📊 Evaluation Results:")
    print(f"BLEU Score: {metrics['bleu_score']:.4f}")
    print(f"ROUGE-1 Score: {metrics['rouge1_score']:.4f}")
    print(f"ROUGE-2 Score: {metrics['rouge2_score']:.4f}")
    print(f"ROUGE-L Score: {metrics['rougeL_score']:.4f}")
    
    # Save detailed results
    results = {
        'metrics': metrics,
        'sample_predictions': [
            {
                'instruction': test_data[i]['instruction'],
                'reference': references[i],
                'prediction': predictions[i],
                'bleu': bleu_scores[i],
                'rouge1': rouge_scores['rouge1'][i]
            }
            for i in range(min(10, len(predictions)))  # Save first 10 examples
        ]
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Detailed results saved to {output_dir}/evaluation_results.json")
    
    return metrics

def compare_with_base_model(test_data_sample=10):
    """Compare fine-tuned model with base model"""
    print(f"\n🆚 Comparing with Base Model (using {test_data_sample} samples)...")
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Load test data
    with open(Config.dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    test_size = int(len(data) * 0.1)
    test_data = data[-test_size:][:test_data_sample]
    
    # Load base model WITHOUT quantization for comparison
    print("Loading base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(Config.model_name, token=hf_token)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        low_cpu_mem_usage=True
    )
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    # Load base model with quantization first
    ft_base_model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=setup_quantization_config(),
        device_map="auto",
        token=hf_token
    )
    ft_model = PeftModel.from_pretrained(ft_base_model, Config.output_dir)
    
    print("\n🔄 Generating responses...")
    
    comparison_results = []
    for i, item in enumerate(test_data):
        prompt = f"<s>[INST] {item['instruction']} [/INST]"
        
        # Base model response
        inputs = base_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            base_outputs = base_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=base_tokenizer.eos_token_id
            )
        
        base_response = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True)
        base_response = base_response.split("[/INST]")[-1].strip()
        
        # Fine-tuned model response
        inputs = base_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            ft_outputs = ft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=base_tokenizer.eos_token_id
            )
        
        ft_response = base_tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
        ft_response = ft_response.split("[/INST]")[-1].strip()
        
        comparison_results.append({
            'question': item['instruction'],
            'reference': item['output'],
            'base_model': base_response,
            'fine_tuned': ft_response
        })
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Q: {item['instruction']}")
        print(f"Reference: {item['output']}")
        print(f"Base Model: {base_response}")
        print(f"Fine-tuned: {ft_response}")
    
    # Save comparison
    with open(os.path.join(Config.output_dir, "model_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✅ Comparison saved to {Config.output_dir}/model_comparison.json")
    
    return comparison_results

def main():
    """Main training function"""
    print("🐝 Starting Mistral 7B Fine-tuning for Beekeeping Dataset")
    print(f"Using {'QLoRA' if Config.use_qlora else 'LoRA'} configuration")
    
    # Pre-flight checks
    print("\n🔍 Running pre-flight checks...")
    
    # Check for dataset
    if not os.path.exists(Config.dataset_path):
        print(f"❌ Dataset not found: {Config.dataset_path}")
        exit(1)
    
    # Check for HF token
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token or hf_token == "hf_your_token_here":
        print("❌ Hugging Face token required!")
        exit(1)
    
    print("✅ Pre-flight checks passed!")
    
    # Initialize wandb if enabled
    if Config.use_wandb:
        wandb.init(
            project="mistral-beekeeping-finetune",
            config=vars(Config),
            name=f"mistral-7b-{'qlora' if Config.use_qlora else 'lora'}-beekeeping"
        )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(tokenizer)
    
    # DON'T tokenize dataset for SFTTrainer - it handles tokenization internally
    print("Dataset prepared for SFTTrainer...")
    
    # Setup LoRA
    print("Setting up LoRA configuration...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Training arguments - FIXED: Use eval_strategy instead of evaluation_strategy
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_epochs,
        max_steps=-1,
        warmup_ratio=Config.warmup_ratio,
        weight_decay=Config.weight_decay,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        eval_steps=Config.eval_steps,
        eval_strategy="steps",  # FIXED: Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,  # Use bfloat16 for L40S
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="wandb" if Config.use_wandb else None,
        run_name="mistral-7b-qlora-beekeeping",
        seed=42,
        data_seed=42,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        save_total_limit=3,
    )
    
    # Initialize trainer - SFTTrainer handles tokenization internally
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        #dataset_text_field="text" Tell SFTTrainer which field contains the text
        # max_seq_length=Config.max_seq_length,
        # tokenizer=tokenizer,
        #packing=False,  # Don't pack sequences
    )
    
    # Start training
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(Config.output_dir)
    
    print("✅ Training completed!")
    print(f"Model saved to: {Config.output_dir}")
    
    # Save training metrics
    if trainer.state.log_history:
        with open(os.path.join(Config.output_dir, "training_log.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    # Evaluate the fine-tuned model
    print("\n🔍 Starting comprehensive evaluation...")
    evaluation_metrics = evaluate_model_performance(model, tokenizer, tokenized_dataset["test"], Config.output_dir)
    
    # Log metrics to wandb
    if Config.use_wandb:
        wandb.log(evaluation_metrics)
    
    return model, tokenizer, evaluation_metrics

def inference_example():
    """Example of how to use the fine-tuned model with evaluation"""
    print("\n🔮 Loading fine-tuned model for inference...")
    
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=setup_quantization_config(),
        device_map="auto",
        token=hf_token
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, Config.output_dir)
    
    # Example inferences with different types of questions
    test_questions = [
        "What are the key factors in maintaining a healthy bee colony?",
        "How does seasonal management affect honey production?",
        "What are the signs of a queenless hive?",
        "How can beekeepers prevent swarming behavior?",
        "What role does propolis play in hive health?"
    ]
    
    print("\n🧪 Testing model responses:")
    for i, question in enumerate(test_questions, 1):
        prompt = f"<s>[INST] {question} [/INST]"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()
        
        print(f"\n--- Test {i} ---")
        print(f"Q: {question}")
        print(f"A: {response}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. GPU required for training.")
        exit(1)
    
    print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check bitsandbytes version
    try:
        import bitsandbytes as bnb
        print(f"📦 BitsAndBytes version: {bnb.__version__}")
    except ImportError:
        print("❌ BitsAndBytes not installed!")
        print("Install with: pip install bitsandbytes>=0.43.2")
        exit(1)
    
    # Run training
    model, tokenizer, metrics = main()
    
    # Run comprehensive evaluation
    print("\n🚀 Running additional evaluations...")
    
    # Compare with base model
    comparison_results = compare_with_base_model(test_data_sample=5)
    
    # Run inference examples
    inference_example()
    
    print("\n✅ All evaluations completed!")
    print(f"📊 Final BLEU Score: {metrics['bleu_score']:.4f}")
    print(f"📊 Final ROUGE-L Score: {metrics['rougeL_score']:.4f}")
    print(f"📁 All results saved in: {Config.output_dir}")
    
    # Summary
    print("\n📋 Evaluation Summary:")
    print("- evaluation_results.json: Detailed metrics and sample predictions")
    print("- model_comparison.json: Side-by-side comparison with base model") 
    print("- training_log.json: Training loss and metrics over time")