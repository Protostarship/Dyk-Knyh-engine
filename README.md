# 🌍 Indigenous Language Translator Engine (ILTE) 🌿

---
##### 📌 Developed by XI TJKT 2 2024/2025 | ❗ Any commercial use or unauthorized exploitation is prohibited
---

```
Release:
- v2.1.0-Beta.2 ALT
- v2.1.1-Alpha.2 ADV
- v2.1.2-Beta.3 ZS
```
---
```
 ██╗  ██╗     ████████╗ ███████╗
 ██║  ██║     ╚══██╔══╝ ██╔════╝
 ██║  ██║        ██║    ███████╗  
 ██║  ██║        ██║    ██╔════╝  
 ██║  ███████╗   ██║    ███████╗
 ╚═╝  ╚══════╝   ╚═╝    ╚══════╝
 -------------------------------
 ILTE - Indigenous Language Translator Engine
```

## 📌 Overview
The **Indigenous Language Translator Engine (ILTE)** now offers **three distinct versions**, each tailored to different translation needs:

- **🌱 ILTE-ALT (Optimized for Speed)** – A lightweight, dictionary-based translator optimized for **fast, low-resource translations**.
- **🧠 ILTE-ZS (Hybrid, Multi-Processing)** – Combines **dictionary-based rules, RBMT, FST, semantic matching, and zero-shot translation** while efficiently handling **large text files**.
- **🧐 ILTE-ADV (AI-Powered, Context-Aware)** – An advanced, AI-driven translation engine that integrates **context awareness, semantic similarity, and zero-shot learning**.

## ✨ Key Features
### **ILTE-ALT - Simple, Fast & Efficient**
- ✅ **Dictionary-Based Lookup** for direct translations.
- ✅ **Basic Stemming for Indonesian (ID) & English (EN)**.
- ✅ **Levenshtein Distance Matching** for closest word lookup.
- ✅ **Automated Confidence Scoring** for accuracy estimation.
- ✅ **Structured DOCX Report Generation**.
- ✅ **Low Memory Usage** – Optimized for lower-end machines.

### **ILTE-ZS - Hybrid, Large-Scale Processing & Efficient**
- ⚡ **Dictionary + RBMT + FST + Semantic Matching + Zero-Shot Translation**.
- ⚖️ **Handles Large Files Efficiently** via **chunking & batch multi-processing**.
- 🛠️ **Optimized Resource Management**, cleans memory and GPU after processing.
- 🔄 **Auto-Parallelized Translation Pipeline**.
- ⏳ **Faster Preprocessing, No Unnecessary Computation**.

### **ILTE-ADV - AI-Powered, Context-Aware & Smarter**
- 🧐 **Contextual Translation using IndoBERT & Sentence Transformers**.
- 🔍 **Zero-Shot Learning for Handling Unknown Words**.
- 📚 **Pattern-Based Learning & Semantic Matching**.
- 🛠️ **Enhanced Translation Confidence Metrics**.
- ⚡ **Leverages GPU Acceleration for Faster Processing**.

## ⚛ Models Used in Each Version

### **🌱 ILTE-ALT (Dictionary-Based)**
| Feature | Model Used |
|---------|-----------|
| **Translation (ID-EN, EN-ID)** | `Helsinki-NLP/opus-mt-id-en`, `Helsinki-NLP/opus-mt-en-id` |
| **Stemming** | `Sastrawi` (Indonesian), `SnowballStemmer` (English) |
| **Fuzzy Matching** | `Levenshtein Distance` |

### **🧠 ILTE-ZS (Hybrid Processing)**
| Feature | Model Used |
|---------|-----------|
| **Dictionary-Based Lookup** | JSON-based dictionary |
| **Rule-Based Translation (RBMT, FST)** | Custom FST Rules |
| **Semantic Similarity** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Zero-Shot Translation** | `facebook/mbart-large-50-many-to-many-mmt` |

### **🧐 ILTE-ADV (AI-Powered)**
| Feature | Model Used |
|---------|-----------|
| **Contextual Embeddings** | `cahya/bert-base-indonesian-1.5G` |
| **Semantic Matching** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Zero-Shot Classification** | `typeform/distilbert-base-uncased-mnli` |
| **Translation (ID-EN, EN-ID)** | `Helsinki-NLP/opus-mt-id-en`, `Helsinki-NLP/opus-mt-en-id` |

## 📊 Comparison Table
| Feature | ILTE-ALT | ILTE-ZS | ILTE-ADV |
|---------|---------|---------|---------|
| **Translation Approach** | Dictionary + Levenshtein | Dictionary + RBMT + FST + Semantic + Zero-Shot | Dictionary + Semantic + Context-Based AI |
| **Processing Speed** | Fast | Moderate (Batch Multi-Processing) | Slower (Context-Aware AI) |
| **Handling Large Files** | Struggles | 🔄 Efficient Chunking & Batch Processing | Slower |
| **Memory Usage** | Low | Moderate | High |
| **Context Awareness** | None | Partial (RBMT & FST) | ✅ Strong (BERT + Semantic Matching) |
| **Idiomatic Expressions** | Limited | Rule-Based (RBMT) | AI-Based |
| **Parallelization** | Minimal | ✅ Thread + Process Pool | DataLoader-Based |
| **Zero-Shot Capability** | No | Yes | Yes |
| **Best Use Case** | Fast, simple translation | Large file handling, scalable processing | High-accuracy, AI-based translation |

## 🗂 How to Use
### **Running ILTE-ALT (Simple Mode)**
```sh
python engine_ALT.py
```
### **Running ILTE-ZS (Hybrid & Efficient Mode)**
```sh
python engine_ZS.py
```
### **Running ILTE-ADV (AI-Powered Mode)**
```sh
python engine_ADV.py
```

---
## 🎯 Conclusion
Choose the version that best suits your needs and contribute to **indigenous language preservation**. 🚀  
- ✅ **ALT:** For lightweight, dictionary-based translations.
- ✅ **ZS:** For handling large files efficiently with hybrid translation techniques.
- ✅ **ADV:** For AI-powered, context-aware translations.

#### 🔗 **Developed for Indigenous Language Preservation** 🌍💡  
#### 📚 **Licensed under GPL v3** – Any commercial use is strictly prohibited.

