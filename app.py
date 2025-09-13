import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import PyPDF2
import fitz  # PyMuPDF

# Hugging Face transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class HoneyDatasetGenerator:
    def __init__(self, pdf_directory: str = "KG", output_file: str = "honey_dataset.jsonl"):
        self.pdf_directory = Path(pdf_directory)
        self.output_file = output_file
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI models
        self.setup_models()
        
    def setup_models(self):
        """Initialize Hugging Face models for content extraction and generation"""
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Question-Answer generation model  
            self.qa_generator = pipeline(
                "text2text-generation",
                model="google/flan-t5-large",
                device=device
            )
            
            # Text summarization for creating outputs
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device
            )
            
            # Content relevance classifier
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device
            )
            
            self.logger.info("✅ All AI models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except:
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e:
                self.logger.error(f"Failed to extract text from {pdf_path}: {e}")
                return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1
            if current_length > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def is_honey_relevant(self, text: str) -> bool:
        """Use AI model to determine if text is honey-related"""
        honey_labels = [
            "honey production and properties",
            "beekeeping and apiculture", 
            "food science and nutrition",
            "chemistry and analysis",
            "medical and therapeutic uses"
        ]
        
        try:
            result = self.classifier(text[:500], honey_labels)
            # Consider relevant if top score is above threshold
            return result['scores'][0] > 0.3
        except:
            # Fallback: simple keyword check
            honey_keywords = ['honey', 'bee', 'nectar', 'apiary', 'crystallization', 'viscosity']
            return any(keyword in text.lower() for keyword in honey_keywords)
    
    def generate_instruction_output_pairs(self, text: str) -> List[Dict[str, str]]:
        """Generate instruction-output pairs using AI models"""
        pairs = []
        
        # Different instruction prompts for variety
        instruction_prompts = [
            "Generate a question about honey properties based on this text:",
            "Create an instruction asking about honey analysis methods from this content:",
            "Form a question about honey production processes from this information:",
            "Generate an instruction about honey quality assessment based on this text:",
            "Create a question about honey chemical composition from this content:",
            "Form an instruction about honey therapeutic properties based on this information:",
            "Generate a question about honey crystallization from this text:",
            "Create an instruction about honey viscosity and rheology from this content:"
        ]
        
        for prompt in instruction_prompts[:3]:  # Limit to 3 per chunk to avoid overloading
            try:
                # Generate instruction
                instruction_input = f"{prompt}\n\nText: {text[:800]}"
                instruction_result = self.qa_generator(
                    instruction_input,
                    do_sample=True,
                    temperature=0.7
                )
                instruction = instruction_result[0]['generated_text'].strip()
                
                # Generate detailed output using summarization + expansion
                summary_result = self.summarizer(
                    text[:1000],
                    do_sample=True
                )
                output = summary_result[0]['summary_text']
                
                # Enhance output with more details
                enhancement_prompt = f"Provide a detailed technical explanation based on this summary about honey: {output}"
                enhanced_result = self.qa_generator(
                    enhancement_prompt,
                    do_sample=True,
                    temperature=0.6
                )
                enhanced_output = enhanced_result[0]['generated_text'].strip()
                
                # Use the original text chunk as output for better quality
                final_output = text[:2000]  # Use raw text instead of generated summaries
                
                if len(instruction) > 10 and len(final_output) > 50:
                    pairs.append({
                        "instruction": instruction,
                        "input": "",
                        "output": final_output
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate pair: {e}")
                continue
        
        return pairs
    
    def create_manual_pairs(self, text: str) -> List[Dict[str, str]]:
        """Create instruction-output pairs using template-based approach as fallback"""
        pairs = []
        
        # Template-based instruction generation
        templates = [
            {
                "instruction": "Explain the physical and chemical properties of honey discussed in research literature.",
                "keywords": ["viscosity", "density", "moisture", "pH", "conductivity"]
            },
            {
                "instruction": "Describe the analytical methods used for honey quality assessment.",
                "keywords": ["analysis", "testing", "quality", "standard", "method"]
            },
            {
                "instruction": "Discuss the factors affecting honey crystallization and texture.",
                "keywords": ["crystallization", "crystal", "texture", "granulation", "solidification"]
            },
            {
                "instruction": "Explain the antimicrobial and therapeutic properties of honey.",
                "keywords": ["antimicrobial", "antibacterial", "therapeutic", "medicinal", "healing"]
            }
        ]
        
        for template in templates:
            if any(keyword in text.lower() for keyword in template["keywords"]):
                # Use the text as output, cleaned up
                clean_text = " ".join(text.split()[:200])  # Limit length
                pairs.append({
                    "instruction": template["instruction"],
                    "input": "",
                    "output": clean_text
                })
        
        return pairs
    
    def process_pdfs(self) -> List[Dict[str, str]]:
        """Process all PDFs and generate training data"""
        all_pairs = []
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            self.logger.error(f"No PDF files found in {self.pdf_directory}")
            return all_pairs
        
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            if not text:
                continue
            
            # Split into chunks
            chunks = self.chunk_text(text, 1500)
            self.logger.info(f"Split into {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                if not self.is_honey_relevant(chunk):
                    continue
                
                self.logger.info(f"Processing relevant chunk {i+1}/{len(chunks)}")
                
                # Try AI generation first
                try:
                    pairs = self.generate_instruction_output_pairs(chunk)
                    all_pairs.extend(pairs)
                except Exception as e:
                    self.logger.warning(f"AI generation failed, using manual approach: {e}")
                    # Fallback to manual pairs
                    manual_pairs = self.create_manual_pairs(chunk)
                    all_pairs.extend(manual_pairs)
        
        self.logger.info(f"Generated {len(all_pairs)} total training pairs")
        return all_pairs
    
    def save_jsonl(self, data: List[Dict[str, str]]):
        """Save data in JSONL format"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.logger.info(f"✅ Saved {len(data)} entries to {self.output_file}")
    
    def run(self):
        """Main execution method"""
        self.logger.info("🚀 Starting AI-powered honey dataset generation...")
        
        # Check directory exists
        if not self.pdf_directory.exists():
            self.logger.error(f"❌ Directory {self.pdf_directory} does not exist!")
            return
        
        # Process PDFs
        training_data = self.process_pdfs()
        
        if not training_data:
            self.logger.error("❌ No training data generated!")
            return
        
        # Save as JSONL
        self.save_jsonl(training_data)
        
        # Print sample
        self.print_sample(training_data)
    
    def print_sample(self, data: List[Dict[str, str]]):
        """Print sample entries"""
        print(f"\n{'='*50}")
        print(f"📊 DATASET SUMMARY")
        print(f"{'='*50}")
        print(f"Total entries: {len(data)}")
        print(f"Output file: {self.output_file}")
        
        print(f"\n{'='*50}")
        print(f"📝 SAMPLE ENTRIES")
        print(f"{'='*50}")
        
        for i, item in enumerate(data[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Instruction: {item['instruction']}")
            print(f"Input: {item['input']}")
            print(f"Output: {item['output'][:200]}...")

def main():
    """Main function"""
    # Configuration - CHANGE THESE PATHS AS NEEDED
    PDF_DIR = "KG"  # Change this to your PDF directory path
    OUTPUT_FILE = "honey_dataset.jsonl"
    
    generator = HoneyDatasetGenerator(PDF_DIR, OUTPUT_FILE)
    generator.run()

if __name__ == "__main__":
    main()