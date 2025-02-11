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
The Indigenous Language Translator Engine is an AI-powered system designed to translate between **Indonesian (ID), English (EN), and Dayak Kenyah (DYK)**. 
This system efficiently processes and translates text using linguistic insights and intelligent dictionary lookup.

## 🚀 Key Features
✅ **Optimized Multi-Step Translation Flow**  
✅ **Basic Stemming Instead of Complex Lemmatization**  
✅ **Dictionary-Based Lookup for Efficient Translation**  
✅ **Automated Confidence Scoring**  
✅ **Detailed DOCX Report Generation**  
✅ **Support for Manual Input and File Uploads**  

## 🔄 Translation Pipeline Breakdown
The translation process follows a **structured multi-step pipeline**:

### 1️⃣ **Preprocessing**
   - Convert input text to lowercase
   - Apply stemming (for Indonesian words)
   - Tokenize words for dictionary lookup

### 2️⃣ **Translation Flow & Intermediary Conversion**
   - **ID → DYK**: Direct dictionary lookup with preprocessing
   - **EN → DYK**: Convert EN → ID first, then process as ID → DYK
   - **DYK → ID**: Direct dictionary mapping
   - **DYK → EN**: Convert DYK → ID first, then process ID → EN

### 3️⃣ **Dictionary Lookup & Matching**
   - Exact match search in the dictionary
   - Stemmed word lookup for better accuracy
   - If no direct match, apply **Levenshtein Distance** for closest word matching

### 4️⃣ **Translation Confidence Calculation**
   - **1.0** → Exact dictionary match
   - **0.9** → Stemmed match found
   - **0.6 - 0.8** → Closest match using Levenshtein distance
   - **0.0 - 0.5** → No reliable match found, fallback to original word

### 5️⃣ **Final Processing & Output Generation**
   - Assemble translated words into a coherent sentence
   - Append metadata (match rate, confidence score, etc.)
   - Generate structured **DOCX Report**

## 📜 How to Use
### Input Options
🔹 **Manual Input**: Type text directly into the CLI.  
🔹 **File Upload**: Provide a file path to process large texts.  

### Running the Translator
#### 1️⃣ Ensure `dictionary.json` is properly formatted.
#### 2️⃣ Run the script:  
   ```sh
   python translator.py
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
🔹 **Expanded Dictionary**: Collaborate with native speakers to refine accuracy.  

## 🎯 Conclusion
The **Indigenous Language Translator Engine** is a robust, scalable, and efficient tool that simplifies translation between **Indonesian, English, and Dayak Kenyah**. By leveraging dictionary-based lookup, intelligent preprocessing, and structured reporting, it ensures high accuracy while maintaining **low performance impact**.

---
🔗 **Developed for Indigenous Language Preservation** 🌍💡


**📌 Translator Developed by XI TJKT 2  |  Any use of this tool for one's own gain is strictly prohibited 📜**
