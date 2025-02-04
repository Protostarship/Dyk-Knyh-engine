# Indigenous Translator Engine Documentation

## Overview
The Indigenous Translator Engine is a Python-based translation system designed to translate between indigenous languages and other languages (such as Indonesian and English). It employs a hybrid approach combining dictionary-based translation, semantic similarity matching, and example-based translation.

## Key Features
- Bidirectional translation support (indigenous ↔ Indonesian/English)
- Hybrid translation approach combining multiple methods
- Confidence scoring and match rate calculation
- Report generation in DOCX format
- Word stemming for improved matching
- Fuzzy word matching using Levenshtein distance

## Prerequisites

### Required Python Packages
- `sentence-transformers`
- `transformers`
- `python-docx`
- `numpy`
- `nltk`
- `python-Levenshtein`

### Required Files
- `dictionary.json`: JSON file containing the translation dictionary
- `examples.docx`: Word document containing example translations

## Technical Architecture

### Model Components

1. **Sentence Transformer**
   - Model: `paraphrase-multilingual-MiniLM-L12-v2`
   - Purpose: Generates semantic embeddings for example-based translation
   - Features: Multilingual support, lightweight architecture

2. **Stemmers**
   - English: Porter Stemmer from NLTK
   - Indonesian: Snowball Stemmer from NLTK
   - Purpose: Reduces words to their root form for improved matching

### Translation Process

The engine follows a multi-step translation process:

1. **Example-based Translation**
   - Computes semantic similarity between input and known examples
   - Uses pre-computed embeddings for efficiency
   - Threshold: 0.8 similarity score for direct example matching

2. **Word-by-word Translation**
   - Attempts exact dictionary matching
   - Falls back to stemmed word matching (90% confidence)
   - Uses Levenshtein distance for fuzzy matching (minimum 60% confidence)

3. **Confidence Calculation**
   - Aggregates confidence scores across translation methods
   - Calculates match rate based on successfully translated words
   - Maximum confidence capped at 90% for word-by-word translation

## Usage

### Initialization
```python
translator = IndigenousTranslator('dictionary.json', 'examples.docx')
```

### Basic Translation
```python
input_text = "Your text here"
source_lang = "id"  # or "en" for English
translation = translator.translate_sentence(input_text, source_lang)
```

### Generating Reports
```python
translator.create_report(input_text, translation, 'translation_report.docx')
```

## Dictionary Format
The `dictionary.json` file should follow this structure:
```json
{
    "word": {
        "indigenous": "indigenous_translation",
        "en": "english_translation",
        "id": "indonesian_translation"
    }
}
```

## Examples Document Format
The `examples.docx` file should contain example translations separated by `||`:
```
Original text || Translation
```

## Performance Metrics

### Confidence Score
- Range: 0.0 to 1.0
- Factors:
  - Exact match: 1.0
  - Stemmed match: 0.9
  - Fuzzy match: Based on Levenshtein distance
  - Example match: Based on semantic similarity

### Match Rate
- Calculation: (Matched words) / (Total words)
- Threshold for matched words: > 0.6 confidence

## Best Practices

1. **Dictionary Maintenance**
   - Keep dictionary entries in lowercase
   - Ensure consistency in translations
   - Regular updates with new vocabulary

2. **Example Management**
   - Include diverse example sentences
   - Focus on common phrases and idioms
   - Maintain clear separation with `||` delimiter

3. **Performance Optimization**
   - Pre-compute embeddings for examples
   - Use appropriate stemming based on source language
   - Implement caching for frequently translated phrases

## Limitations

1. **Translation Accuracy**
   - Limited by dictionary completeness
   - Dependent on example quality
   - May struggle with complex grammar structures

2. **Performance Considerations**
   - Initial loading time due to model loading
   - Memory usage with large example sets
   - Computation time for semantic similarity

3. **Language Support**
   - Currently limited to English and Indonesian as source languages
   - Requires separate stemmer implementation for additional languages

## Future Improvements

1. **Enhanced Features**
   - Grammar rule implementation
   - Context-aware translation
   - Additional language support
   - Automated dictionary updates

2. **Technical Optimizations**
   - Improved caching mechanisms
   - Parallel processing for batch translations
   - Memory optimization for large-scale use

## Support

For issues and improvements, please:
1. Check the dictionary format
2. Verify example document structure
3. Ensure all dependencies are correctly installed
4. Monitor confidence scores for unexpected results
