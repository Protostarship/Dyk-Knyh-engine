# 🌍 Indigenous Language Translator Engine (ILTE) 🌿

<div align="center">
  <img src="https://github.com/Protostarship/Dyk-Knyh-engine/blob/main/animated-tiles.svg" width="100%" alt="banner"/>
</div>
<div align="center">
    <img src="https://github.com/Protostarship/Dyk-Knyh-engine/blob/main/bg.jpg" style="width: 100%; height: 90px; object-fit: cover;" alt="Profile Banner">
</div>

##### 📌 Developed by XI TJKT 2 | 2024/2025 | ❗ Any commercial use or unauthorized exploitation is prohibited
---

Release:
- [v2.1.0-Beta.2 ALT](https://github.com/Protostarship/Dyk-Knyh-engine/releases/tag/v2.1.0)
- [v2.1.1-Alpha.2 ADV](https://github.com/Protostarship/Dyk-Knyh-engine/releases/tag/v2.1.0)
- [v2.1.2-Beta.3 ZS](https://github.com/Protostarship/Dyk-Knyh-engine/releases/tag/v2.1.2-Beta.3)
- [v3.0.0-Alpha.3 ATI](https://github.com/Protostarship/Dyk-Knyh-engine/releases/tag/v3.0.0-Alpha.3)
  
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
The **Indigenous Language Translator Engine (ILTE)** now offers **four distinct versions**, each tailored to different translation needs:

- **🌱 ILTE-ALT (Optimized for Speed)** – A lightweight, dictionary-based translator optimized for **fast, low-resource translations**.
- **🧠 ILTE-ZS (Hybrid, Multi-Processing)** – Combines **dictionary-based rules, RBMT, FST, semantic matching, and zero-shot translation** while efficiently handling **large text files**.
- **🧐 ILTE-ADV (AI-Powered, Context-Aware)** – An advanced, AI-driven translation engine that integrates **context awareness, semantic similarity, and zero-shot learning**.
- **🔮 ILTE-ATI (Advanced Attention & Iterative Processing)** – The most sophisticated version with **hierarchical normalization, iterative refinement, attention-based translation, and multi-level candidate selection**.

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
- 🧠 **Contextual Translation using IndoBERT & Sentence Transformers**.
- 🔍 **Zero-Shot Learning for Handling Unknown Words**.
- 📚 **Pattern-Based Learning & Semantic Matching**.
- 🛠️ **Enhanced Translation Confidence Metrics**.
- ⚡ **Leverages GPU Acceleration for Faster Processing**.

### **ILTE-ATI - Attention-Based, Iterative & Highly Adaptive**
- ✨ **Hierarchical Normalization for Better Preprocessing**.
- 🔄 **Iterative Translation for Context Awareness**.
- 📚 **Attention-Based Translation for Multi-Level Candidate Generation**.
- ⚖️ **Refined Confidence Scoring & Adaptive Refinement**.
- ✅ **Full Formatting Preservation in DOCX Reports**.
- 🚀 **Optimized for Dynamic, Multi-Stage Translation Processes**.

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

### **🔮 ILTE-ATI v3-Alpha.3 (Attention-Based & Iterative Processing)**
| Feature | Model Used |
|---------|-----------|
| **Hierarchical Normalization** | Regex + Dynamic Stemming |
| **Contextual Translation** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Iterative Processing** | Multi-Level Candidate Refinement |
| **Translation (ID-DYK, DYK-ID)** | Enhanced Dictionary Lookup + Semantic Matching |

## 📊 Comparison Table
| Feature | ILTE-ALT | ILTE-ZS | ILTE-ADV | ILTE-ATI |
|---------|---------|---------|---------|---------------------|
| **Translation Approach** | Dictionary | Hybrid | AI-Based | Attention-Based + Iterative |
| **Processing Speed** | Fast | Moderate | Slower | Balanced |
| **Handling Large Files** | Struggles | Efficient Chunking | Slower | Optimized Processing |
| **Memory Usage** | Low | Moderate | High | Optimized |
| **Context Awareness** | None | Partial | Strong | 🔮 Very Strong |
| **Idiomatic Expressions** | Limited | Rule-Based | AI-Based | AI + Attention |
| **Parallelization** | Minimal | Yes | DataLoader | Thread + Process Pool |
| **Zero-Shot Capability** | No | Yes | Yes | Yes |
| **Best Use Case** | Fast translation | Large text processing | Context-Aware | High-Accuracy, AI-Powered |

## 📚 How to Use
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
### **Running ILTE-ATI (Advanced Iterative Attention Engine)**
```sh
python engine_ATI.py
```

---
## 🎯 Conclusion
Choose the version that best suits your needs and contribute to **indigenous language preservation**. 🚀  
- ✅ **ALT:** For lightweight, dictionary-based translations.
- ✅ **ZS:** For handling large files efficiently with hybrid translation techniques.
- ✅ **ADV:** For AI-powered, context-aware translations.
- ✅ **ATI** For advanced attention towards content and context.

#### 🔗 **Developed for Indigenous Language Preservation** 🌍💡  
#### 📚 **Licensed under GPL v3** – Any commercial use is strictly prohibited.

