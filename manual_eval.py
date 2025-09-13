#!/usr/bin/env python3
"""
Simple manual evaluation - just generate responses for you to judge
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAINED_MODEL_PATH = "./mistral-7b-beekeeping-lora"
HF_TOKEN = getenv("hg")

os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

def setup_model():
    """Load the fine-tuned model"""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
        token=HF_TOKEN
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
    
    return model, tokenizer

def generate_response(model, tokenizer, question):
    """Generate response"""
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
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
    return response.split("[/INST]")[-1].strip()

def main():
    """Interactive evaluation"""
    print("🔍 Manual Model Evaluation")
    print("Ask beekeeping questions to test your model!")
    print("Type 'quit' to exit\n")
    
    model, tokenizer = setup_model()
    
    # Sample questions to get started
    sample_questions = [
        "What causes bee colonies to swarm?",
        "How do you treat Nosema disease?",
        "What's the difference between worker bees and drones?",
        "When should you add honey supers?",
        "How do you identify a good queen bee?"
    ]
    
    print("Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "-"*50)
    
    while True:
        question = input("\n🤔 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        print("\n🤖 Model response:")
        print("-" * 40)
        response = generate_response(model, tokenizer, question)
        print(response)
        print("-" * 40)
        
        # Simple rating
        while True:
            try:
                rating = input("\n📊 Rate this response (1-5, or press Enter to skip): ").strip()
                if not rating:
                    break
                rating = int(rating)
                if 1 <= rating <= 5:
                    print(f"✅ You rated this response: {rating}/5")
                    break
                else:
                    print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
    
    print("\n👋 Thanks for testing the model!")

if __name__ == "__main__":
    main()