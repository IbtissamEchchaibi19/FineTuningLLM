import os
import json
import torch
import wandb
import numpy as np
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
import warnings

warnings.filterwarnings("ignore")

# Download NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration
class Config:
    def __init__(self):
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.dataset_path = "training_dataset.jsonl"
        self.output_dir = "./mistral-7b-beekeeping-lora"
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.use_qlora = True
        self.batch_size = 2
        self.gradient_accumulation_steps = 8
        self.learning_rate = 5e-5
        self.num_epochs = 2
        self.max_seq_length = 1024
        self.warmup_ratio = 0.03
        self.weight_decay = 0.001
        self.logging_steps = 25
        self.save_steps = 200
        self.eval_steps = 200
        self.use_wandb = True

Config = Config()

def setup_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def format_instruction_dataset(examples):
    formatted_texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        formatted_text = f"<s>[INST] {instruction.strip()} [/INST] {output.strip()}</s>"
        formatted_texts.append(formatted_text)
    return {"text": formatted_texts}

def load_and_prepare_dataset(tokenizer):
    if not os.path.exists(Config.dataset_path):
        print(f"Dataset file not found: {Config.dataset_path}")
        exit(1)

    data = []
    with open(Config.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                if 'text' in item:
                    data.append({'text': item['text']})
                elif 'instruction' in item and 'output' in item:
                    data.append({'instruction': item['instruction'], 'output': item['output']})

    if 'instruction' in data[0]:
        instructions = [item['instruction'] for item in data]
        outputs = [item['output'] for item in data]
        dataset = Dataset.from_dict({"instruction": instructions, "output": outputs})
        dataset = dataset.map(format_instruction_dataset, batched=True, remove_columns=dataset.column_names)
    else:
        texts = [item['text'] for item in data]
        dataset = Dataset.from_dict({"text": texts})

    dataset = dataset.train_test_split(test_size=0.15, seed=42)
    return dataset

def setup_model_and_tokenizer():
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token or hf_token == "your_hf_token":
        print("Hugging Face token not set!")
        exit(1)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name, token=hf_token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=setup_quantization_config(),
        device_map="auto",
        trust_remote_code=True,
        token=hf_token
    )

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def setup_lora_config():
    return LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=Config.target_modules,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="gaussian"
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = predictions[..., :-1, :].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}

def evaluate_model_responses(model, tokenizer, test_questions=None):
    if not test_questions:
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
            }
        ]

    model.eval()
    results = []

    for item in test_questions:
        prompt = f"<s>[INST] {item['instruction']} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        expected_topics = item.get('expected_topics', [])
        topics_covered = sum([1 for t in expected_topics if t.lower() in response.lower()])
        coverage_score = topics_covered / len(expected_topics) if expected_topics else 0

        results.append({
            "question": item['instruction'],
            "response": response,
            "topic_coverage": f"{topics_covered}/{len(expected_topics)}",
            "coverage_score": coverage_score,
            "response_length": len(response.split())
        })

    avg_coverage = np.mean([r['coverage_score'] for r in results])
    avg_length = np.mean([r['response_length'] for r in results])

    return results, {"avg_coverage": avg_coverage, "avg_length": avg_length}

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. GPU required for training.")
        exit(1)

    if Config.use_wandb:
        wandb.init(
            project="mistral-beekeeping-finetune-v2",
            config=vars(Config),
            name="mistral-7b-qlora-beekeeping-improved"
        )

    model, tokenizer = setup_model_and_tokenizer()
    dataset = load_and_prepare_dataset(tokenizer)

    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_epochs,
        warmup_ratio=Config.warmup_ratio,
        weight_decay=Config.weight_decay,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        eval_steps=Config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if Config.use_wandb else None,
        seed=42,
        data_seed=42,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        save_total_limit=2,
        eval_accumulation_steps=4,
        prediction_loss_only=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=Config.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_kwargs={"add_special_tokens": False, "append_concat_token": False}
    )

    base_results, base_metrics = evaluate_model_responses(model, tokenizer)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(Config.output_dir)
    ft_results, ft_metrics = evaluate_model_responses(model, tokenizer)

    improvement = ft_metrics['avg_coverage'] - base_metrics['avg_coverage']
    results = {
        "base_model_metrics": base_metrics,
        "fine_tuned_metrics": ft_metrics,
        "improvement": improvement,
        "base_model_results": base_results,
        "fine_tuned_results": ft_results
    }

    with open(os.path.join(Config.output_dir, "evaluation_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    if Config.use_wandb:
        wandb.log({
            "final_coverage_improvement": improvement,
            "final_coverage_score": ft_metrics['avg_coverage'],
            "final_avg_length": ft_metrics['avg_length']
        })

    return model, tokenizer, results

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. GPU required.")
        exit(1)

    model, tokenizer, results = main()
    print("Training and evaluation completed.")
    print(f"Results saved in: {Config.output_dir}")
    print(f"Coverage improvement: {results['improvement']:+.3f}")
