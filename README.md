# 🌍 Indigenous Language Translator Engine 🌿

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
The Indigenous Language Translator Engine is an AI-powered system designed to translate between **Indonesian (ID), English (EN), and Dayak Kenyah (DYK)** using **Sastrawi Stemmer for Indonesian**, **Snowball Stemmer for English**, and **optimized dictionary-based lookup**.

## 🚀 Key Features (v1.0.0)
### ✅ **Optimized Multi-Step Translation Flow** (EN ↔ ID ↔ DYK)  
### ✅ **Dual Stemming Support** – **Sastrawi Stemmer** (ID) & **Snowball Stemmer** (EN)  
### ✅ **Dictionary-Based Lookup for Efficient Translation**  
### ✅ **Levenshtein Distance Approximation for Closest Matches**  
### ✅ **Automated Confidence Scoring & Translation Accuracy Calculation**  
### ✅ **DOCX Report Generation with Structured Headers & Footers**  
### ✅ **Support for Manual Text Input and File-Based Translations**  

## 🖥️ System Requirements
To ensure smooth operation of ILTE, the following system requirements are recommended:

### 🔹 **Minimum Requirements**
- **CPU**: Intel Core i3 (4th Gen) / AMD Ryzen 3
- **RAM**: 4GB
- **Storage**: 500MB free disk space
- **OS**: Windows 10, macOS 10.13+, or Ubuntu 18.04+
- **Python Version**: 3.8+
- **Internet**: Required for downloading NLP models

### 🔹 **Recommended Requirements**
- **CPU**: Intel Core i5 (8th Gen) / AMD Ryzen 5 or higher
- **RAM**: 8GB or higher (for better translation speed)
- **Storage**: 1GB free disk space
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python Version**: 3.10+
- **Internet**: Required for downloading and updating NLP models

## 🔄 Translation Process Breakdown
The translation process follows a **structured multi-step pipeline**:

### 1️⃣ **Preprocessing**
   - Convert input text to lowercase.
   - Apply **Sastrawi Stemmer** for Indonesian words.
   - Apply **Snowball Stemmer** for English words.
   - Tokenize words for dictionary lookup.

### 2️⃣ **Translation Flow & Processing**
   - **ID → DYK**: Stem text, perform dictionary lookup, apply closest match if needed.
   - **EN → DYK**: Convert EN → ID first, then process ID → DYK.
   - **DYK → ID**: Direct dictionary mapping with closest match handling.
   - **DYK → EN**: Convert DYK → ID first, then process ID → EN.

### 3️⃣ **Dictionary Lookup & Matching**
   - **Exact match search** in the dictionary.
   - **Stemmed word lookup** for improved accuracy.
   - **Levenshtein Distance Approximation** for closest word matching.

### 4️⃣ **Translation Confidence Calculation**
   - **1.0** → Exact dictionary match.
   - **0.9** → Stemmed match found.
   - **0.6 - 0.8** → Closest match using Levenshtein distance.
   - **0.0 - 0.5** → No reliable match found, fallback to original word.

### 5️⃣ **Final Processing & Output Generation**
   - Assemble translated words into a structured format.
   - Append metadata (match rate, confidence score, and translation accuracy).
   - Generate structured **DOCX Report**.

## 📜 How to Use
### Input Options
🔹 **Manual Input**: Type text directly into the CLI.  
🔹 **File Upload**: Provide a file path for bulk translation.  

### Running the Translator
#### 1️⃣ Ensure `dictionary.json` is properly formatted.
#### 2️⃣ Run the script:  
   ```sh
   python engine_alt.py
   ```
#### 3️⃣ Select input type (`file` or `text`).
#### 4️⃣ Provide the **source language** and **target language**.
#### 5️⃣ Receive **translation output** and **DOCX report**.

## 📂 Managing the Dictionary
The translation relies on a **JSON dictionary** stored in the format:
```json
{
    "makan": "ngakan",
    "minum": "nyuip"
}
```
### Adding New Words
📌 **Ensure lowercase formatting** for better accuracy.  
📌 **Use precise and validated indigenous translations**.  

## 📑 Report Structure
The generated **DOCX Report** includes:
- 📌 **Header**: Performance Score, Translation Rate, and Confidence.
- 📜 **Body**: Only the translation results.
- 🔻 **Footer**: Translation origin, original and target word counts.

## 🛠 Future Enhancements
🔹 **Context-Aware Translations**: Improve sentence-level understanding.  
🔹 **GUI Interface**: Introduce a user-friendly graphical interface.  
🔹 **Expanded Dictionary**: Crowdsourcing indigenous language data.  

## 🎯 Conclusion
The **Indigenous Language Translator Engine** v1.0.0 is an advanced yet lightweight translation tool designed for **linguists, educators, and researchers**. With **dual stemming techniques, intelligent dictionary lookups, and structured reporting**, ILTE ensures **high accuracy while maintaining low performance impact**.  

---
🔗 **Developed for Indigenous Language Preservation** 🌍💡  
📜 **Licensed under GPL v3 – Any use of this tool for one's own gain is strictly prohibited** 🔥


**📌 Translator Developed by XI TJKT 2  |  Any use of this tool for one's own gain is strictly prohibited 📜**
