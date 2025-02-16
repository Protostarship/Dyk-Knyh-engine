
---

# ILTE Translation Engine Models Comparison Report

This report provides a detailed side‐by‐side comparison of four ILTE translation engine models:

- **Engine_ADV**
- **Engine_ALT**
- **Engine_ATI**
- **Engine_ZS**

Each model has evolved over time, adding new features and optimizations. The following sections summarize their key differences and strengths.

---

## Summary Table

| **Feature**                    | **Engine_ADV**                                    | **Engine_ALT**                                    | **Engine_ATI**                                    | **Engine_ZS**                                     |
|--------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| **Dictionary Used**            | External ID-DYK dictionary                        | External ID-DYK dictionary                        | External ID-DYK dictionary                        | External ID-DYK dictionary                        |
| **Malay Dictionary Usage**     | Not used                                          | Not used                                          | Removed                                           | Integrated Malay cognate detection                |
| **External Translation Pipelines** | Helsinki-NLP for en→id and id→en                  | Helsinki-NLP for en→id and id→en                  | Helsinki-NLP for en→id (for candidate generation) | Helsinki-NLP for en→id; Zero-shot fallback (MBart)  |
| **Context Window**             | Configurable (default: 3)                         | Minimal; word-by-word matching                    | Broader context (default: 10)                      | Adaptive context processing                       |
| **Preprocessing Methods**      | Basic stemming (Sastrawi & Snowball)              | Basic stemming (Sastrawi & Snowball)              | Hierarchical normalization using regex & stemming | Unicode normalization and advanced token cleaning |
| **Similarity Matching**        | SentenceTransformer (MiniLM)                      | Levenshtein distance matching                     | SentenceTransformer-based semantic matching        | SentenceTransformer with zero-shot fallback         |
| **Batch Processing & Parallelization** | Yes (DataLoader for batching)                   | Limited/None                                      | Yes (Multi-threaded & multi-process)              | Yes (Multi-threaded & multi-process)              |
| **Zero-shot Translation**      | Not implemented                                   | Not implemented                                   | Not implemented                                   | Implemented (MBart-based zero-shot translation)     |
| **Caching**                    | Not implemented                                   | Not implemented                                   | JSON-based caching with expiry                    | JSON-based caching with locking                   |
| **Handling of Unknown Words**  | Context-based similarity matching                 | Levenshtein-based closest match                   | Iterative translation with dynamic candidate generation | Combined RBMT, FST, semantic matching, and zero-shot fallback |
| **Report Generation**          | Basic DOCX report with overall metrics            | Basic DOCX report                                 | Detailed DOCX report with preserved formatting and metrics | Detailed DOCX report with refined final translations and formatting |
| **Resource Management**        | Uses GPU for BERT and SentenceTransformer          | CPU-based processing                              | Optimized resource management with caching and batching | Advanced resource management with explicit GPU memory handling |

---

## Detailed Comparison

### Engine_ADV
- **Overview**:  
  Uses advanced context encoding with a SentenceTransformer to precompute dictionary embeddings.  
- **Preprocessing**:  
  Relies on both Sastrawi and Snowball stemmers.  
- **Translation Approach**:  
  Uses Helsinki-NLP pipelines for English-to-Indonesian and Indonesian-to-English translation.  
- **Batch Processing**:  
  Implements batching via PyTorch’s DataLoader and processes text in batches.
- **Limitations**:  
  Lacks robust candidate generation for unknown words and doesn’t implement caching.

---

### Engine_ALT
- **Overview**:  
  A simpler engine that performs direct dictionary lookup and uses Levenshtein distance for finding the closest match.  
- **Preprocessing**:  
  Utilizes Snowball and Sastrawi stemmers for normalization.
- **Translation Approach**:  
  Processes text word-by-word without a broader context, which may result in less accurate translations.
- **Limitations**:  
  Minimal parallel processing and no caching; less context-aware.

---

### Engine_ATI
- **Overview**:  
  Improves upon earlier models by incorporating a broader context window and iterative translation.  
- **Preprocessing**:  
  Implements hierarchical normalization (lowercasing, regex-based cleaning, and stemming) without hardcoded mappings.  
- **Translation Approach**:  
  Dynamically generates candidates (e.g. via the Helsinki-NLP en→id translator) when direct dictionary lookups fail.  
- **Optimization**:  
  Uses multi-threading and multi-processing with JSON-based caching.
- **Report Generation**:  
  Generates detailed DOCX reports that preserve the original text formatting.
- **Strengths**:  
  Iterative processing refines the translation until stabilization.

---

### Engine_ZS
- **Overview**:  
  The most advanced version, enhancing the ATI model with rule-based machine translation (RBMT), FST improvements, and a zero-shot fallback.  
- **Preprocessing**:  
  Builds upon hierarchical normalization while integrating additional rules for idiomatic expressions.
- **Translation Approach**:  
  Combines dictionary lookup, FST rules, semantic matching, and zero-shot translation (using MBart) to handle unknown words robustly.
- **Optimization**:  
  Incorporates comprehensive resource management, caching, and parallel processing.
- **Report Generation**:  
  Produces a refined DOCX report that includes final translations with preserved paragraph breaks.
- **Strengths**:  
  Most robust and context-aware, suitable for production environments with challenging inputs.

---

## Conclusion

- **Engine_ADV** is best for applications requiring context-encoded semantic matching and batch processing, though it lacks fallback mechanisms.
- **Engine_ALT** offers a straightforward dictionary-based solution with Levenshtein matching but does not leverage context.
- **Engine_ATI** enhances translation quality through iterative processing, dynamic candidate generation, and robust caching.
- **Engine_ZS** combines all previous improvements with rule-based refinements and a zero-shot fallback for the most robust, context-aware translation.

This report outlines the evolution of the ILTE engines and provides a clear comparison of each model’s features and performance characteristics.

---
