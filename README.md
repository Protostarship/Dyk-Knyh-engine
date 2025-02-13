# 🌍 Indigenous Language Translator Engine (ILTE) 🌿

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
The **Indigenous Language Translator Engine (ILTE)** offers **two distinct versions** designed for different levels of translation complexity:

- **🌱 ILTE-ALT (Optimized for Speed)** – A lightweight, dictionary-based translator optimized for **fast, low-resource translations**.
- **🧠 ILTE-ADV (AI-Powered, Context-Aware)** – An advanced, AI-driven translation engine that integrates **context awareness, semantic similarity, and zero-shot learning**.

## ✨ Key Features
### **ILTE-ALT - Simple, Fast & Efficient**
- ✅ **Dictionary-Based Lookup** for direct translations.
- ✅ **Basic Stemming for Indonesian (ID) & English (EN)**.
- ✅ **Levenshtein Distance Matching** for closest word lookup.
- ✅ **Automated Confidence Scoring** for accuracy estimation.
- ✅ **Structured DOCX Report Generation**.
- ✅ **Low Memory Usage** – Optimized for lower-end machines.

### **ILTE-ADV - AI-Powered, Context-Aware & Smarter**
- 🧠 **Contextual Translation using IndoBERT & Sentence Transformers**.
- 🔍 **Zero-Shot Learning for Handling Unknown Words**.
- 📖 **Pattern-Based Learning & Semantic Matching**.
- 🔄 **Enhanced Translation Confidence Metrics**.
- ⚡ **Leverages GPU Acceleration for Faster Processing**.

## ⚛ Models Used in Each Version

### **🌱 ILTE-ALT (Dictionary-Based)**
| Feature | Model Used |
|---------|-----------|
| **Translation (ID-EN, EN-ID)** | `Helsinki-NLP/opus-mt-id-en`, `Helsinki-NLP/opus-mt-en-id` |
| **Stemming** | `Sastrawi` (Indonesian), `SnowballStemmer` (English) |
| **Fuzzy Matching** | `Levenshtein Distance` |

### **🧠 ILTE-ADV (AI-Powered)**
| Feature | Model Used |
|---------|-----------|
| **Contextual Embeddings** | `cahya/bert-base-indonesian-1.5G` |
| **Semantic Matching** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Zero-Shot Classification** | `typeform/distilbert-base-uncased-mnli` |
| **Translation (ID-EN, EN-ID)** | `Helsinki-NLP/opus-mt-id-en`, `Helsinki-NLP/opus-mt-en-id` |

## 🔄 Translation Flow Breakdown
### **1️⃣ Preprocessing**
- **ALT:** Lowercasing, stemming, tokenization.
- **ADV:** Context-aware tokenization, embedding generation, and sentence structure analysis.

### **2️⃣ Translation Flow & Processing**
#### **ILTE-ALT (Fast Dictionary-Based Approach)**
- **ID → DYK**: Direct dictionary lookup.
- **EN → DYK**: Converts **EN → ID**, then **ID → DYK**.
- **DYK → EN**: Converts **DYK → ID**, then **ID → EN**.

#### **ILTE-ADV (AI-Powered Context-Aware Approach)**
- **ID → DYK**: Uses **contextual embeddings** & **semantic matching**.
- **EN → DYK**: Uses **zero-shot classification + IndoBERT embeddings**.
- **DYK → EN**: Adapts to **sentence structures & known translation patterns**.

### **3️⃣ Dictionary Lookup & Matching**
- **ALT:** Exact match ➔ Stemmed match ➔ Levenshtein Distance.
- **ADV:** **Semantic Similarity ➔ Contextual Matching ➔ Pattern Recognition**.

### **4️⃣ Confidence Calculation**
- **ALT:** Based on dictionary and fuzzy matching.
- **ADV:** Uses **contextual confidence scoring, IndoBERT embeddings, and AI classifiers**.

### **5️⃣ Output Generation**
- **Both versions generate structured DOCX reports**.
- **ADV additionally tracks pattern memory & learning rates**.

## 📂 How to Use
### **Choose a Version Based on Your Needs**
#### ✔️ Use **ILTE-ALT** for lightweight, dictionary-based translations.
#### ✔️ Use **ILTE-ADV** for AI-powered, context-aware translations.

### **Running ILTE-ALT (Simple Mode)**
```sh
python engine_ALT.py
```

### **Running ILTE-ADV (AI-Powered Mode)**
```sh
python engine_ADV.py
```

### **Selecting Input Type**
👉 **Manual Input**: Type text directly into the CLI.
👉 **File Upload**: Provide a file path for batch translation.

## 📁 Managing the Dictionary
- The dictionary is stored in **JSON format**.
- Format:
```json
{
  "apa"   : "inu",
  "kemana": "kenpi"
}
```

### Adding New Words
#### 📌 **Ensure lowercase formatting**.
#### 📌 **Use precise and validated indigenous translations**.

## 📚 Report Structure
The **DOCX Report** includes:
- **Header**: Performance Score, Translation Rate, and Confidence.
- **Body**: Only the translation results.
- **Footer**: Original and target word counts.

## 💻 System Requirements
### **ILTE-ALT - Minimal System Requirements**
- **CPU**: Intel Core i3 / AMD Ryzen 3
- **RAM**: 4GB+
- **Storage**: 500MB free space
- **Python**: 3.8+

### **ILTE-ADV - AI-Powered, Requires More Resources**
- **CPU**: Intel Core i5 / AMD Ryzen 5
- **RAM**: 8GB+ (16GB Recommended)
- **Storage**: 1GB+ free space
- **Python**: 3.10+
- **GPU Acceleration (Recommended)**

## 🛠 Future Enhancements
### 🔹 **ALT:** Optimize performance, add basic semantic matching.
### 🔹 **ADV:** Improve AI logic, better zero-shot classification.

## 🏆 Conclusion
Choose the version that best suits your needs and contribute to **indigenous language preservation**. 🚀  

---
🔗 **Developed for Indigenous Language Preservation** 🌍💡  
📚 **Licensed under GPL v3** – Any commercial use is strictly prohibited.  

📌 **Note from XI TJKT 2 Development Team** 
- 💡 **Any use of our translation engine for one's own gain is strictly prohibited!**


__**"Never takes advantages of others just for your own gains."**__
