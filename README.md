# 🌍 Indigenous Language Translator Engine (ILTE) 🌿

```
 ██╗██╗     ████████╗███████╗
 ██║██║     ╚══██╔══╝██╔════╝
 ██║██║        ██║   █████╗  
 ██║██║        ██║   ██╔══╝  
 ██║███████╗   ██║   ███████╗
 ╚═╝╚══════╝   ╚═╝   ╚══════╝
 ILTE - Indigenous Language Translator Engine
```

## 📌 Overview
The Indigenous Language Translator Engine (ILTE) offers **two distinct versions** to cater to different user needs:

- **🌱 ILTE-ALT (Beta Release)** – A lightweight, efficient translation engine focused on **basic dictionary-based** translations.
- **🧠 ILTE-ADV (Alpha Release)** – An advanced version that leverages **AI-driven context-awareness and semantic matching** for smarter translations.

## 🚀 Key Features
### **ILTE-ALT (Beta) - Lightweight & Efficient**
- ✅ **Basic Stemming for ID & EN** (Sastrawi for ID, Snowball for EN)  
- ✅ **Dictionary-Based Lookup** for direct and closest matches  
- ✅ **Levenshtein Distance** for fuzzy matching  
- ✅ **Automated Confidence Scoring**  
- ✅ **Structured DOCX Report Generation**  
- ✅ **Low Memory Usage** – Ideal for lower-end machines  

### **ILTE-ADV (Alpha) - Smart & Context-Aware**
- 🧠 **Contextual Translation with IndoBERT & Sentence Transformers**  
- 🔎 **Zero-Shot Learning for Unknown Words**  
- 📖 **Pattern-Based Learning** – Adapts to text structures  
- 🔄 **Semantic Similarity Matching** – Finds best translations beyond dictionary  
- 📊 **Enhanced Translation Confidence Metrics**  
- ⚡ **Requires Higher Memory & Processing Power** – Ideal for researchers and professionals  

## 🔄 Translation Pipeline Breakdown
### **1️⃣ Preprocessing**
- **ALT**: Converts text to lowercase, applies stemming, tokenizes words.
- **ADV**: Performs **context-aware tokenization, embedding generation, and text structure analysis**.

### **2️⃣ Translation Flow & Processing**
#### **ILTE-ALT (Beta)**
- **ID → DYK**: Basic dictionary lookup with stemming.
- **EN → DYK**: Converts **EN → ID** first, then **ID → DYK**.
- **DYK → EN**: Converts **DYK → ID** first, then **ID → EN**.

#### **ILTE-ADV (Alpha)**
- **ID → DYK**: Uses **contextual embeddings** & **semantic matching**.
- **EN → DYK**: Uses **AI-driven conversion pipelines**.
- **DYK → EN**: Adapts to **sentence structures and known translation patterns**.

### **3️⃣ Dictionary Lookup & Matching**
- **ALT**: Exact match → Stemmed match → Levenshtein Distance.
- **ADV**: **Semantic Similarity** → **Pattern Recognition** → **AI-Based Contextual Matching**.

### **4️⃣ Translation Confidence Calculation**
- **ALT**: Based on dictionary and fuzzy matching.
- **ADV**: Integrates **contextual confidence scoring, IndoBERT embeddings, and AI classifiers**.

### **5️⃣ Final Processing & Output Generation**
- **Both versions generate structured DOCX reports**.
- **ADV additionally tracks pattern memory & learning rates**.

## 📜 How to Use
### **Choose a Version Based on Your Needs**
#### 1️⃣ **Use ILTE-ALT (Beta) for lightweight, fast translations with minimal system requirements.**  
#### 2️⃣ **Use ILTE-ADV (Alpha) for smarter, AI-powered translations that improve over time.**  

### **Running ILTE-ALT (Beta) – Fast & Simple**
```sh
python engine_ALT.py
```

### **Running ILTE-ADV (Alpha) – AI-Enhanced**
```sh
python engine_ADV.py
```

### **Selecting Input Type**
🔹 **Manual Input**: Type text directly into the CLI.  
🔹 **File Upload**: Provide a file path for bulk translation.  

## 📂 Managing the Dictionary
The translation relies on a **JSON dictionary** stored in the format:
```json
{
    "makan": "ngakan",
    "minum": "nyuip"
}
```
### Adding New Words
#### 📌 **Ensure lowercase formatting** for better accuracy.  
#### 📌 **Use precise and validated indigenous translations**.  

## 📑 Report Structure
The generated **DOCX Report** includes:
- 📌 **Header**: Performance Score, Translation Rate, and Confidence.
- 📜 **Body**: Only the translation results.
- 🔻 **Footer**: Translation origin, original and target word counts.

## 🖥️ System Requirements
### **ILTE-ALT (Beta) - Minimal System Impact**
- **CPU**: Intel Core i3 / AMD Ryzen 3  
- **RAM**: 4GB+  
- **Storage**: 500MB free space  
- **Python**: 3.8+  

### **ILTE-ADV (Alpha) - AI-Powered, Requires More Resources**
- **CPU**: Intel Core i5 / AMD Ryzen 5  
- **RAM**: 8GB+ (16GB Recommended)  
- **Storage**: 1GB+ free space  
- **Python**: 3.10+  
- **GPU Acceleration (Optional, but Recommended)**  

## 🛠 Future Enhancements
### 🔹 **ALT**: Additional optimization for better performance.  
### 🔹 **ADV**: Improved AI logic, better zero-shot classification, enhanced pattern learning.  

## 🎯 Conclusion
The **Indigenous Language Translator Engine** (ILTE) now offers two pathways:  
- **ILTE-ALT**: A simple, dictionary-based translator for fast and effective results.  
- **ILTE-ADV**: A powerful AI-driven translator that continuously learns and improves.  

Choose the version that best suits your needs, and contribute to the development of **indigenous language preservation**. 🚀  

---
🔗 **Developed for Indigenous Language Preservation** 🌍💡  
📜 **Licensed under GPL v3 – Any use of this tool for one's own gain is strictly prohibited** 🔥



**📌 Translator Developed by XI TJKT 2  |  Any use of this tool for one's own gain is strictly prohibited 📜**
