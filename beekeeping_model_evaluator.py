import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import nltk

nltk.download('punkt', quiet=True)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TRAINED_MODEL_PATH = "./mistral-7b-beekeeping-lora"
TEST_DATASET_PATH = "test_dataset.jsonl"
HF_TOKEN = os.getenv("hf")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


class BeekeepingModelEvaluator:
    def __init__(self, model_name, trained_model_path, test_dataset_path, hf_token):
        self.model_name = model_name
        self.trained_model_path = trained_model_path
        self.test_dataset_path = test_dataset_path
        self.hf_token = hf_token
        self.base_model = None
        self.ft_model = None
        self.tokenizer = None

    def create_sample_test_data(self):
        test_questions = [
            {"instruction": "What equipment is essential for a beginner beekeeper?",
             "expected_topics": ["hive tool", "smoker", "protective gear", "frames", "foundation"]},
            {"instruction": "How do you identify if your hive has been robbed by other bees?",
             "expected_topics": ["dead bees", "wax cappings", "reduced honey stores", "aggressive behavior"]},
            {"instruction": "What are the main differences between Italian and Carniolan bee breeds?",
             "expected_topics": ["temperament", "brood pattern", "honey production", "wintering ability"]},
            {"instruction": "How should you prepare your hives for winter in cold climates?",
             "expected_topics": ["insulation", "ventilation", "food stores", "entrance reducers", "windbreaks"]},
            {"instruction": "What causes chalkbrood disease and how is it treated?",
             "expected_topics": ["fungal infection", "Ascosphaera apis", "moisture control", "hive ventilation"]},
            {"instruction": "When is the best time to split a strong hive?",
             "expected_topics": ["spring", "drone cells", "queen cells", "population", "nectar flow"]},
            {"instruction": "How do you assess the quality of a queen bee?",
             "expected_topics": ["egg laying pattern", "brood pattern", "pheromones", "worker behavior"]},
            {"instruction": "What plants are best for supporting bee populations throughout the season?",
             "expected_topics": ["diverse bloom times", "native plants", "clover", "wildflowers", "fruit trees"]},
            {"instruction": "How do you safely remove bees from a structure without killing them?",
             "expected_topics": ["live removal", "bee vacuum", "relocation", "comb removal", "prevention"]},
            {"instruction": "What are the signs that indicate your hive is preparing to swarm?",
             "expected_topics": ["queen cells", "crowded conditions", "reduced foraging", "clustering"]}
        ]
        with open(self.test_dataset_path, 'w', encoding='utf-8') as f:
            for item in test_questions:
                f.write(json.dumps(item) + '\n')
        return test_questions

    def load_test_data(self):
        if not os.path.exists(self.test_dataset_path):
            return self.create_sample_test_data()
        data = []
        with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def setup_quantization(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def load_models(self):
        if not os.path.exists(self.trained_model_path):
            raise FileNotFoundError(f"Trained model not found: {self.trained_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.setup_quantization(),
            device_map="auto",
            token=self.hf_token
        )

        ft_base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.setup_quantization(),
            device_map="auto",
            token=self.hf_token
        )
        self.ft_model = PeftModel.from_pretrained(ft_base_model, self.trained_model_path)

    def generate_response(self, model, instruction, max_tokens=150):
        prompt = f"<s>[INST] {instruction} [/INST]"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("[/INST]")[-1].strip()

    def evaluate_relevance(self, response, expected_topics):
        response_lower = response.lower()
        covered = sum(1 for topic in expected_topics if topic.lower() in response_lower)
        return covered / len(expected_topics) if expected_topics else 0

    def evaluate_models(self, test_data):
        results = {'base_model': {'responses': [], 'relevance_scores': []},
                   'fine_tuned': {'responses': [], 'relevance_scores': []}}
        detailed_results = []

        for item in test_data:
            instruction = item['instruction']
            expected_topics = item.get('expected_topics', [])

            base_resp = self.generate_response(self.base_model, instruction)
            ft_resp = self.generate_response(self.ft_model, instruction)

            base_score = self.evaluate_relevance(base_resp, expected_topics)
            ft_score = self.evaluate_relevance(ft_resp, expected_topics)

            results['base_model']['responses'].append(base_resp)
            results['base_model']['relevance_scores'].append(base_score)

            results['fine_tuned']['responses'].append(ft_resp)
            results['fine_tuned']['relevance_scores'].append(ft_score)

            detailed_results.append({
                'question': instruction,
                'expected_topics': expected_topics,
                'base_model': {'response': base_resp, 'relevance_score': base_score},
                'fine_tuned': {'response': ft_resp, 'relevance_score': ft_score}
            })

        return results, detailed_results

    def calculate_metrics(self, results):
        metrics = {}
        for model_name in results:
            scores = results[model_name]['relevance_scores']
            metrics[model_name] = {
                'avg_relevance': np.mean(scores),
                'std_relevance': np.std(scores),
                'num_samples': len(scores)
            }
        improvement = metrics['fine_tuned']['avg_relevance'] - metrics['base_model']['avg_relevance']
        return metrics, improvement

    def save_results(self, metrics, improvement, detailed_results):
        results_dir = os.path.join(self.trained_model_path, "test_evaluation")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "evaluation_summary.json"), "w") as f:
            json.dump({'metrics': metrics, 'improvement': improvement}, f, indent=2)
        with open(os.path.join(results_dir, "detailed_comparison.json"), "w") as f:
            json.dump(detailed_results, f, indent=2)
        return results_dir

    def run_evaluation(self):
        test_data = self.load_test_data()
        self.load_models()
        results, detailed_results = self.evaluate_models(test_data)
        metrics, improvement = self.calculate_metrics(results)
        self.save_results(metrics, improvement, detailed_results)
        return metrics, improvement, detailed_results


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. GPU recommended.")
    evaluator = BeekeepingModelEvaluator(MODEL_NAME, TRAINED_MODEL_PATH, TEST_DATASET_PATH, HF_TOKEN)
    metrics, improvement, detailed = evaluator.run_evaluation()
    print("Evaluation completed")
    print("Metrics:", metrics)
    print("Improvement:", improvement)


if __name__ == "__main__":
    main()
