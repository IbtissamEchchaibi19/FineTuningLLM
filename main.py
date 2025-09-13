import json
import random
import re
from typing import List, Dict, Any, Tuple
import nltk
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline
import logging

# Download required NLTK data - Updated for newer NLTK versions
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# New punkt_tab tokenizer for newer NLTK versions
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Also download the updated POS tagger if available
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng')
    except:
        pass  # Fall back to the original tagger

class DataAugmentator:
    def __init__(self):
        """Initialize the data augmentation pipeline."""
        self.setup_logging()
        self.load_models()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load required models for augmentation and quality control."""
        self.logger.info("Loading models...")
        
        # Sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Paraphrasing model
        try:
            self.paraphraser = pipeline(
                "text2text-generation",
                model="Vamsi/T5_Paraphrase_Paws",
                device=-1  # Use CPU
            )
        except:
            self.logger.warning("Could not load paraphrasing model. Will use synonym replacement only.")
            self.paraphraser = None
            
        self.logger.info("Models loaded successfully!")
        
    def load_data(self, file_path: str) -> List[Dict]:
        """Load JSONL data from file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        self.logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
        
    def save_data(self, data: List[Dict], file_path: str):
        """Save data to JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        self.logger.info(f"Saved {len(data)} examples to {file_path}")
        
    def get_synonyms(self, word: str, pos_tag: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        
        # Map POS tags to WordNet POS
        pos_map = {
            'NN': wordnet.NOUN, 'NNS': wordnet.NOUN, 'NNP': wordnet.NOUN, 'NNPS': wordnet.NOUN,
            'VB': wordnet.VERB, 'VBD': wordnet.VERB, 'VBG': wordnet.VERB, 'VBN': wordnet.VERB, 'VBP': wordnet.VERB, 'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ, 'JJR': wordnet.ADJ, 'JJS': wordnet.ADJ,
            'RB': wordnet.ADV, 'RBR': wordnet.ADV, 'RBS': wordnet.ADV
        }
        
        wn_pos = pos_map.get(pos_tag)
        if not wn_pos:
            return []
            
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and len(synonym.split()) == 1:
                    synonyms.add(synonym)
                    
        return list(synonyms)[:3]  # Limit to 3 synonyms
        
    def synonym_replacement(self, text: str, replacement_rate: float = 0.1) -> str:
        """Replace words with synonyms."""
        try:
            words = nltk.word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
        except Exception as e:
            self.logger.warning(f"Tokenization failed: {e}. Returning original text.")
            return text
        
        n_replacements = max(1, int(len(words) * replacement_rate))
        word_indices = list(range(len(words)))
        random.shuffle(word_indices)
        
        replacements_made = 0
        for i in word_indices:
            if replacements_made >= n_replacements:
                break
                
            word, pos = pos_tags[i]
            if len(word) > 2 and word.isalpha():  # Only replace meaningful words
                synonyms = self.get_synonyms(word, pos)
                if synonyms:
                    words[i] = random.choice(synonyms)
                    replacements_made += 1
                    
        return ' '.join(words)
        
    def paraphrase_text(self, text: str) -> str:
        """Generate paraphrase using T5 model."""
        if not self.paraphraser:
            return self.synonym_replacement(text)
            
        try:
            # Limit text length for processing
            if len(text) > 500:
                text = text[:500]
                
            input_text = f"paraphrase: {text}"
            result = self.paraphraser(input_text, max_length=len(text.split()) + 50, num_return_sequences=1)
            paraphrased = result[0]['generated_text']
            
            # Clean up the result
            paraphrased = paraphrased.replace("paraphrase: ", "").strip()
            return paraphrased if paraphrased else text
            
        except Exception as e:
            self.logger.warning(f"Paraphrasing failed: {e}. Using synonym replacement.")
            return self.synonym_replacement(text)
            
    def back_translate_simulate(self, text: str) -> str:
        """Simulate back-translation by applying multiple transformations."""
        # Simple simulation: synonym replacement + minor restructuring
        transformed = self.synonym_replacement(text, 0.15)
        
        # Add some structural variations
        try:
            sentences = nltk.sent_tokenize(transformed)
            if len(sentences) > 1 and random.random() < 0.3:
                # Occasionally shuffle sentences
                random.shuffle(sentences)
                transformed = ' '.join(sentences)
        except Exception as e:
            self.logger.warning(f"Sentence tokenization failed: {e}")
            # Fall back to original transformed text
            
        return transformed
        
    def augment_text_field(self, text: str, technique: str) -> str:
        """Apply specific augmentation technique to text."""
        if not text or len(text.strip()) < 10:
            return text
            
        if technique == "paraphrase":
            return self.paraphrase_text(text)
        elif technique == "synonym":
            return self.synonym_replacement(text)
        elif technique == "back_translate":
            return self.back_translate_simulate(text)
        else:
            return text
            
    def augment_example(self, example: Dict, technique: str) -> Dict:
        """Augment a single example."""
        augmented = example.copy()
        
        # Common text fields to augment (adjust based on your data structure)
        text_fields = ['text', 'instruction', 'input', 'output', 'response', 'question', 'answer', 'content']
        
        for field in text_fields:
            if field in augmented and isinstance(augmented[field], str):
                original_text = augmented[field]
                augmented_text = self.augment_text_field(original_text, technique)
                augmented[field] = augmented_text
                
        # Add metadata
        augmented['augmentation_technique'] = technique
        augmented['is_augmented'] = True
        
        return augmented
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.5  # Default neutral similarity
            
    def quality_filter(self, original: Dict, augmented: Dict, 
                      min_similarity: float = 0.7, max_similarity: float = 0.95) -> bool:
        """Filter augmented examples based on quality metrics."""
        
        # Get main text field for comparison
        text_fields = ['text', 'instruction', 'input', 'output', 'response', 'question', 'answer', 'content']
        
        original_text = ""
        augmented_text = ""
        
        for field in text_fields:
            if field in original and isinstance(original[field], str):
                original_text = original[field]
                augmented_text = augmented.get(field, "")
                break
                
        if not original_text or not augmented_text:
            return False
            
        # Check if augmented text is too similar or too different
        similarity = self.calculate_semantic_similarity(original_text, augmented_text)
        
        if similarity < min_similarity or similarity > max_similarity:
            return False
            
        # Check for basic quality issues
        if len(augmented_text) < len(original_text) * 0.5:  # Too short
            return False
            
        if len(augmented_text) > len(original_text) * 2:  # Too long
            return False
            
        # Check for repetitive text
        words = augmented_text.lower().split()
        if len(set(words)) < len(words) * 0.3:  # Too repetitive
            return False
            
        return True
        
    def remove_duplicates(self, data: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """Remove near-duplicate examples."""
        self.logger.info("Removing duplicates...")
        
        # Extract text for comparison
        text_fields = ['text', 'instruction', 'input', 'output', 'response', 'question', 'answer', 'content']
        
        texts = []
        for item in data:
            text = ""
            for field in text_fields:
                if field in item and isinstance(item[field], str):
                    text = item[field]
                    break
            texts.append(text)
            
        # Calculate embeddings
        embeddings = self.sentence_model.encode(texts)
        
        # Find duplicates
        keep_indices = []
        for i, embedding in enumerate(embeddings):
            is_duplicate = False
            for j in keep_indices:
                similarity = cosine_similarity([embedding], [embeddings[j]])[0][0]
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep_indices.append(i)
                
        filtered_data = [data[i] for i in keep_indices]
        self.logger.info(f"Removed {len(data) - len(filtered_data)} duplicates")
        
        return filtered_data
        
    def augment_dataset(self, input_file: str, output_file: str):
        """Main augmentation pipeline."""
        self.logger.info("Starting data augmentation pipeline...")
        
        # Load original data
        original_data = self.load_data(input_file)
        target_size = len(original_data) * 2
        
        # Augmentation techniques to use
        techniques = ["paraphrase", "synonym", "back_translate"]
        
        augmented_data = original_data.copy()
        
        # Calculate how many examples we need to generate
        examples_needed = target_size - len(original_data)
        
        self.logger.info(f"Generating {examples_needed} augmented examples...")
        
        generated_count = 0
        for i, original_example in enumerate(original_data):
            if generated_count >= examples_needed:
                break
                
            # Try each technique
            for technique in techniques:
                if generated_count >= examples_needed:
                    break
                    
                try:
                    augmented_example = self.augment_example(original_example, technique)
                    
                    # Quality filtering
                    if self.quality_filter(original_example, augmented_example):
                        augmented_data.append(augmented_example)
                        generated_count += 1
                        
                        if generated_count % 100 == 0:
                            self.logger.info(f"Generated {generated_count}/{examples_needed} examples")
                            
                except Exception as e:
                    self.logger.warning(f"Augmentation failed for example {i}: {e}")
                    continue
                    
        self.logger.info(f"Generated {generated_count} augmented examples")
        
        # Remove duplicates
        final_data = self.remove_duplicates(augmented_data)
        
        # Save results
        self.save_data(final_data, output_file)
        
        self.logger.info(f"Augmentation complete!")
        self.logger.info(f"Original dataset: {len(original_data)} examples")
        self.logger.info(f"Final dataset: {len(final_data)} examples")
        self.logger.info(f"Augmentation ratio: {len(final_data) / len(original_data):.2f}x")
        
        return final_data

# Usage example
if __name__ == "__main__":
    # Initialize augmentator
    augmentator = DataAugmentator()
    
    # Run augmentation
    augmented_data = augmentator.augment_dataset(
        input_file="honey_dataset.jsonl",
        output_file="augmented_data.jsonl"
    )
    
    print(f"Augmentation completed! Check 'augmented_data.jsonl' for results.")