import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import PyPDF2
import fitz
from transformers import pipeline
import torch

class HoneyInstructionDatasetBuilder:
    def __init__(self, pdf_directory: str = "KG", output_file: str = "honey_dataset.jsonl"):
        self.pdf_directory = Path(pdf_directory)
        self.output_file = output_file
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.setup_models()
        
    def setup_models(self):
        device = 0 if torch.cuda.is_available() else -1
        try:
            self.qa_generator = pipeline("text2text-generation", model="google/flan-t5-large", device=device)
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
            self.logger.info("All AI models loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return text
        except:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return "".join(page.extract_text() for page in reader.pages)
            except Exception as e:
                self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
                return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        words = text.split()
        chunks, current_chunk, current_length = [], [], 0
        for word in words:
            current_length += len(word) + 1
            if current_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [word], len(word)
            else:
                current_chunk.append(word)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    
    def is_honey_relevant(self, text: str) -> bool:
        labels = ["honey production and properties", "beekeeping and apiculture", 
                  "food science and nutrition", "chemistry and analysis", 
                  "medical and therapeutic uses"]
        try:
            result = self.classifier(text[:500], labels)
            return result['scores'][0] > 0.3
        except:
            keywords = ['honey', 'bee', 'nectar', 'apiary', 'crystallization', 'viscosity']
            return any(k in text.lower() for k in keywords)
    
    def generate_instruction_output_pairs(self, text: str) -> List[Dict[str, str]]:
        pairs = []
        prompts = [
            "Generate a question about honey properties based on this text:",
            "Create an instruction asking about honey analysis methods from this content:",
            "Form a question about honey production processes from this information:"
        ]
        for prompt in prompts:
            try:
                instruction_input = f"{prompt}\n\nText: {text[:800]}"
                instruction = self.qa_generator(instruction_input, do_sample=True, temperature=0.7)[0]['generated_text'].strip()
                output = text[:2000]
                if len(instruction) > 10 and len(output) > 50:
                    pairs.append({"instruction": instruction, "input": "", "output": output})
            except Exception as e:
                self.logger.warning(f"Failed to generate pair: {e}")
        return pairs
    
    def create_manual_pairs(self, text: str) -> List[Dict[str, str]]:
        pairs = []
        templates = [
            {"instruction": "Explain the physical and chemical properties of honey.", "keywords": ["viscosity", "density", "moisture", "pH", "conductivity"]},
            {"instruction": "Describe the analytical methods used for honey quality assessment.", "keywords": ["analysis", "testing", "quality", "standard", "method"]},
            {"instruction": "Discuss the factors affecting honey crystallization and texture.", "keywords": ["crystallization", "crystal", "texture", "granulation", "solidification"]},
            {"instruction": "Explain the antimicrobial and therapeutic properties of honey.", "keywords": ["antimicrobial", "antibacterial", "therapeutic", "medicinal", "healing"]}
        ]
        for t in templates:
            if any(k in text.lower() for k in t["keywords"]):
                clean_text = " ".join(text.split()[:200])
                pairs.append({"instruction": t["instruction"], "input": "", "output": clean_text})
        return pairs
    
    def process_pdfs(self) -> List[Dict[str, str]]:
        all_pairs = []
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            self.logger.error(f"No PDF files found in {self.pdf_directory}")
            return all_pairs
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
            chunks = self.chunk_text(text, 1500)
            for chunk in chunks:
                if not self.is_honey_relevant(chunk):
                    continue
                try:
                    pairs = self.generate_instruction_output_pairs(chunk)
                    all_pairs.extend(pairs)
                except Exception as e:
                    self.logger.warning(f"AI generation failed, using manual pairs: {e}")
                    all_pairs.extend(self.create_manual_pairs(chunk))
        self.logger.info(f"Generated {len(all_pairs)} training pairs")
        return all_pairs
    
    def save_jsonl(self, data: List[Dict[str, str]]):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.logger.info(f"Saved {len(data)} entries to {self.output_file}")
    
    def run(self):
        if not self.pdf_directory.exists():
            self.logger.error(f"Directory {self.pdf_directory} does not exist!")
            return
        training_data = self.process_pdfs()
        if not training_data:
            self.logger.error("No training data generated!")
            return
        self.save_jsonl(training_data)

def main():
    generator = HoneyInstructionDatasetBuilder("KG", "honey_dataset.jsonl")
    generator.run()

if __name__ == "__main__":
    main()
