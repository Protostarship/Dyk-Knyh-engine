import json
import numpy as np
from Levenshtein import distance as lev_distance
from nltk.stem import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import torch
from docx import Document
import os
from collections import defaultdict

class EnhancedIndigenousTranslator:
    def __init__(self, json_path):
        # Load base components
        self.dictionary = self.load_json(json_path)
        factory = StemmerFactory()
        self.id_stemmer = factory.create_stemmer()
        self.en_stemmer = SnowballStemmer("english")
        
        # Initialize translation models
        self.en_id_translator = pipeline(
            "translation_en_to_id", model="Helsinki-NLP/opus-mt-en-id"
        )
        self.id_en_translator = pipeline(
            "translation_id_to_en", model="Helsinki-NLP/opus-mt-id-en"
        )
        
        # Initialize Indonesian BERT for context understanding
        self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
        self.context_model = AutoModel.from_pretrained("indolem/indobert-base-uncased")
        
        # Initialize sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        
        # Initialize zero-shot classifier
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli"
        )
        
        # Translation tracking
        self.confidence = 1.0
        self.translations = []
        self.pattern_memory = defaultdict(dict)
        self.context_cache = {}
        
    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
            
    def get_contextual_embedding(self, text):
        """Get contextual embedding using IndoBERT."""
        if text in self.context_cache:
            return self.context_cache[text]
            
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.context_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        self.context_cache[text] = embedding
        return embedding
        
    def find_semantic_matches(self, word, context):
        """Find semantically similar matches using sentence transformers."""
        word_embedding = self.sentence_model.encode(word)
        context_embedding = self.sentence_model.encode(context)
        
        candidates = []
        for dict_word in self.dictionary:
            dict_embedding = self.sentence_model.encode(dict_word)
            word_similarity = np.dot(word_embedding, dict_embedding) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(dict_embedding)
            )
            context_similarity = np.dot(context_embedding, dict_embedding) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(dict_embedding)
            )
            combined_score = 0.7 * word_similarity + 0.3 * context_similarity
            candidates.append((dict_word, combined_score))
            
        return sorted(candidates, key=lambda x: x[1], reverse=True)
        
    def analyze_patterns(self, text, window_size=3):
        """Analyze text patterns for consistent translations."""
        words = text.split()
        patterns = {}
        
        for i in range(len(words)):
            window = words[max(0, i-window_size):min(len(words), i+window_size+1)]
            context = " ".join(window)
            
            # Get contextual understanding
            context_embedding = self.get_contextual_embedding(context)
            
            # Check for known patterns
            pattern_key = tuple(window)
            if pattern_key in self.pattern_memory:
                patterns[words[i]] = self.pattern_memory[pattern_key]
                
        return patterns
        
    def preprocess_text(self, text, lang):
        """Enhanced preprocessing with context preservation."""
        text = text.lower()
        words = text.split()
        
        if lang == "id":
            processed = [self.id_stemmer.stem(word) for word in words]
        elif lang == "en":
            processed = [self.en_stemmer.stem(word) for word in words]
        else:
            processed = words
            
        # Preserve original-to-processed mapping
        mapping = dict(zip(words, processed))
        return " ".join(processed), mapping
        
    def find_closest_match(self, word, context=""):
        """Enhanced closest match finding with context awareness."""
        # First try exact dictionary match
        if word in self.dictionary:
            return word, 1.0
            
        # Try semantic matching
        semantic_matches = self.find_semantic_matches(word, context)
        if semantic_matches and semantic_matches[0][1] > 0.8:
            return semantic_matches[0]
            
        # Fallback to Levenshtein distance
        candidates = list(self.dictionary.keys())
        distances = [(c, lev_distance(word, c)) for c in candidates]
        closest = min(distances, key=lambda x: x[1])
        max_len = max(len(word), len(closest[0]))
        confidence = 1 - (closest[1] / max_len)
        
        return closest[0], confidence
        
    def translate_word(self, word, context=""):
        """Enhanced word translation with context awareness."""
        # Check pattern memory
        patterns = self.analyze_patterns(context)
        if word in patterns:
            return patterns[word], 0.9
            
        # Try dictionary lookup
        if word in self.dictionary:
            return self.dictionary[word], 1.0
            
        # Try finding closest match
        closest, confidence = self.find_closest_match(word, context)
        if confidence > 0.6:
            translation = self.dictionary.get(closest, word)
            # Update pattern memory
            self.pattern_memory[tuple(context.split())][word] = translation
            return translation, confidence
            
        # Zero-shot classification as last resort
        candidates = list(self.dictionary.keys())
        result = self.zero_shot(
            context,
            candidate_labels=candidates,
            hypothesis_template="This text contains the word {}."
        )
        
        if result['scores'][0] > 0.7:
            return self.dictionary[result['labels'][0]], result['scores'][0]
            
        return word, 0.0
        
    def translate_text(self, text, source_lang, target_lang):
        """Enhanced text translation with context awareness."""
        self.confidence = 1.0
        
        # Handle language pipeline
        if source_lang == "en" and target_lang == "dyk":
            text = self.en_id_translator(text, max_length=512)[0]["translation_text"]
        elif source_lang == "dyk" and target_lang == "en":
            text = self.translate_text(text, "dyk", "id")
            text = self.id_en_translator(text, max_length=512)[0]["translation_text"]
            return text
            
        # Preprocess text
        processed_text, word_mapping = self.preprocess_text(text, source_lang)
        words = processed_text.split()
        
        # Translate with context
        translated = []
        total_words = len(words)
        matched_words = 0
        
        for i, word in enumerate(words):
            # Get context window
            context_window = words[max(0, i-2):min(len(words), i+3)]
            context = " ".join(context_window)
            
            # Translate word
            trans, confidence = self.translate_word(word, context)
            translated.append(trans)
            self.confidence *= confidence
            if confidence > 0:
                matched_words += 1
                
        # Update statistics
        self.translations.append((
            text,
            " ".join(translated),
            self.confidence,
            total_words,
            len(translated)
        ))
        self.match_rate = matched_words / total_words if total_words > 0 else 0
        
        return " ".join(translated)
        
    def create_report(self, filename):
        """Enhanced report creation with additional metrics."""
        doc = Document()
        doc.add_heading("Translation Report", 0)
        
        # Calculate metrics
        avg_confidence = sum(conf for _, _, conf, _, _ in self.translations) / len(self.translations)
        total_words = sum(orig_count for _, _, _, orig_count, _ in self.translations)
        total_translations = sum(trans_count for _, _, _, _, trans_count in self.translations)
        
        # Add header
        header = doc.sections[0].header
        header_paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        header_paragraph.text = (
            f"Performance Score: {avg_confidence:.1%} | "
            f"Translation Rate: {self.match_rate:.1%} | "
            f"Pattern Matches: {len(self.pattern_memory)}"
        )
        
        # Add translations
        for original, translated, confidence, original_word_count, target_word_count in self.translations:
            doc.add_paragraph(f"Original: {original}")
            doc.add_paragraph(f"Translated: {translated}")
            doc.add_paragraph(f"Confidence: {confidence:.1%}")
            doc.add_paragraph(f"Words: {original_word_count} → {target_word_count}")
            doc.add_paragraph("\n")
            
        # Add footer
        footer = doc.sections[0].footer
        footer_paragraph = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        footer_paragraph.text = (
            f"Total Words Processed: {total_words} | "
            f"Total Translations: {total_translations} | "
            f"Average Confidence: {avg_confidence:.1%}"
        )
        
        doc.add_page_break()
        doc.save(filename)

if __name__ == "__main__":
    translator = EnhancedIndigenousTranslator("dictionary_alt.json")
    
    choice = input("Enter 'file' to load from file or 'text' to input manually: ")
    if choice == 'file':
        file_path = input("Enter file path: ")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        else:
            print("File not found.")
            exit()
    else:
        text = input("Enter text to translate: ")
    
    source_lang = input("Enter source language (id/en/dyk): ")
    target_lang = input("Enter target language (id/en/dyk): ")
    
    translation = translator.translate_text(text, source_lang, target_lang)
    translator.create_report("translation_report.docx")
    
    print(f"Translation: {translation}")
    print(f"Confidence: {translator.confidence:.1%}")