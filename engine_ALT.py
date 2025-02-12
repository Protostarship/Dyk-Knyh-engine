import json
import numpy as np
from Levenshtein import distance as lev_distance
from nltk.stem import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from transformers import pipeline
from docx import Document
import os

class IndigenousTranslator:
    def __init__(self, json_path):
        self.dictionary = self.load_json(json_path)
        factory = StemmerFactory()
        self.id_stemmer = factory.create_stemmer()
        self.en_stemmer = SnowballStemmer("english")
        self.en_id_translator = pipeline(
            "translation_en_to_id", model="Helsinki-NLP/opus-mt-en-id"
        )
        self.id_en_translator = pipeline(
            "translation_id_to_en", model="Helsinki-NLP/opus-mt-id-en"
        )
        self.confidence = 1.0
        self.translations = []

    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def preprocess_text(self, text, lang):
        text = text.lower()
        if lang == "id":
            return self.id_stemmer.stem(text)
        elif lang == "en":
            return " ".join([self.en_stemmer.stem(word) for word in text.split()])
        return text

    def find_closest_match(self, word):
        candidates = list(self.dictionary.keys())
        if not candidates:
            return word, 0.0

        distances = [(c, lev_distance(word, c)) for c in candidates]
        closest = min(distances, key=lambda x: x[1])
        max_len = max(len(word), len(closest[0]))
        return closest[0], 1 - (closest[1] / max_len)

    def translate_word(self, word):
        if word in self.dictionary:
            return self.dictionary[word], 1.0
        else:
            closest, confidence = self.find_closest_match(word)
            if confidence > 0.6:
                return self.dictionary.get(closest, word), confidence
            return word, 0.0

    def translate_text(self, text, source_lang, target_lang):
        self.confidence = 1.0

        if source_lang == "en" and target_lang == "dyk":
            text = self.en_id_translator(text, max_length=512)[0]["translation_text"]
        elif source_lang == "dyk" and target_lang == "en":
            text = self.translate_text(text, "dyk", "id")
            text = self.id_en_translator(text, max_length=512)[0]["translation_text"]
            return text

        words = text.split()
        translated = []
        total_words = len(words)
        matched_words = 0
        
        for word in words:
            trans, confidence = self.translate_word(word)
            translated.append(trans)
            self.confidence *= confidence
            if confidence > 0:
                matched_words += 1
        
        self.translations.append((text, " ".join(translated), self.confidence, total_words, len(translated)))
        self.match_rate = matched_words / total_words if total_words > 0 else 0
        return " ".join(translated)

    def create_report(self, filename):
        doc = Document()
        doc.add_heading("Translation Report", 0)
        
        avg_confidence = sum(conf for _, _, conf, _, _ in self.translations) / len(self.translations)
        
        header = doc.sections[0].header
        header_paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
        header_paragraph.text = f"Performance Score: {avg_confidence:.1%} | Translation Rate: {self.match_rate:.1%}"
        
        for original, translated, confidence, original_word_count, target_word_count in self.translations:
            doc.add_paragraph(f"Original: {original}")
            doc.add_paragraph(f"Translated: {translated}")
            doc.add_paragraph(f"Confidence: {confidence:.1%}")
            doc.add_paragraph("\n")
        
        footer = doc.sections[0].footer
        footer_paragraph = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        footer_paragraph.text = f"Translation Origin: {original_word_count} words | Target: {target_word_count} words"
        
        doc.add_page_break()
        doc.save(filename)

if __name__ == "__main__":
    translator = IndigenousTranslator("dictionary_alt.json")
    
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

