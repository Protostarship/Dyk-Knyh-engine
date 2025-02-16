# 🖥️ System Requirements for ILTE Engines

## 🚀 Overview
This document provides the **minimum and recommended system requirements** for running all four ILTE engine models efficiently. While **ILTE-ALT** is lightweight, **ILTE-ITE (v3.0.0-Alpha.3)** requires higher computational power due to its advanced processing capabilities.

---

## 🏆 **General Requirements**
| Component   | Minimum Requirement | Recommended Requirement |
|------------|---------------------|-------------------------|
| **OS** | Windows 10 / Linux Ubuntu 20.04+ | Windows 11 / Linux Ubuntu 22.04 |
| **Python Version** | 3.8+ | 3.10+ |
| **Storage** | 2GB Free Space | 5GB SSD/NVMe |
| **Network** | Stable internet for model downloads | High-speed internet (Fiber) |

---

## 🌱 **ILTE-ALT v2.1.0-Beta.2 (Dictionary-Based Engine)**
| Component   | Minimum Requirement | Recommended Requirement |
|------------|---------------------|-------------------------|
| **CPU** | Intel Core i3 / AMD Ryzen 3 | Intel Core i5 / AMD Ryzen 5 |
| **RAM** | 4GB | 8GB |
| **GPU** | Not required | NVIDIA GTX 1050+ for acceleration |
| **Dependencies** | `torch`, `transformers`, `nltk`, `Sastrawi`, `python-docx` |
| **Processing Threads** | 2 | 4 |
| **Batch Processing** | No | Yes |

✅ **Ideal for:** Fast translations with minimal hardware and direct dictionary-based translation.

---

## 🧠 **ILTE-ZS v2.1.2-Beta.3 (Hybrid Engine with RBMT & FST)**
| Component   | Minimum Requirement | Recommended Requirement |
|------------|---------------------|-------------------------|
| **CPU** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / AMD Ryzen 7 |
| **RAM** | 8GB | 16GB |
| **GPU** | NVIDIA GTX 1050+ | NVIDIA RTX 2060+ |
| **Dependencies** | `sentence-transformers`, `Levenshtein`, `malaya`, `fasttext-wheel`, `protobuf` |
| **Processing Threads** | 4 | 8 |
| **Batch Processing** | Yes | Yes (Optimized with multi-threading) |
| **Parallelization** | Partial | Yes (Thread + Process Pool) |

✅ **Ideal for:** Efficient batch translation with **multi-processing, dictionary fallback, and semantic matching**.

---

## 🔍 **ILTE-ADV v2.1.1-Alpha.2 (AI-Powered, Context-Aware Engine)**
| Component   | Minimum Requirement | Recommended Requirement |
|------------|---------------------|-------------------------|
| **CPU** | Intel Core i7 / AMD Ryzen 7 | Intel Core i9 / AMD Ryzen 9 |
| **RAM** | 16GB | 32GB |
| **GPU** | NVIDIA RTX 2060+ | NVIDIA RTX 3090 / A100 |
| **Dependencies** | `sentence-transformers`, `transformers`, `bitsandbytes`, `fasttext`, `scipy`, `numpy` |
| **Processing Threads** | 8 | 12 |
| **Batch Processing** | Yes | Yes (Optimized with DataLoader) |
| **Parallelization** | Yes (Thread & Process Pool) | Yes (CUDA Acceleration) |

✅ **Ideal for:** **Context-aware, AI-driven translations with BERT-based embedding models**.

---

## ✨ **ILTE-ATI v3.0.0-Alpha.3 (Iterative Attention-Based Engine)**
| Component   | Minimum Requirement | Recommended Requirement |
|------------|---------------------|-------------------------|
| **CPU** | Intel Core i5 / AMD Ryzen 5 | AMD Ryzen 7 / Intel Core i7 |
| **RAM** | 16GB | 32GB |
| **GPU** | NVIDIA RTX 3050+ | NVIDIA RTX 4050+ |
| **Dependencies** | `transformers`, `sentence-transformers`, `malaya`, `protobuf`, `torch`, `threadpoolctl`, `chardet` |
| **Processing Threads** | 8 | 12+ |
| **Batch Processing** | Yes (Optimized with adaptive batching) | Yes (Efficient multi-level parallelization) |
| **Parallelization** | Yes (Hybrid multi-threading & CUDA) | Yes (Full CUDA & Multi-GPU Scaling) |
| **Cache Optimization** | JSON-based caching | JSON + Memory-based caching with locking |

✅ **Ideal for:** **High-accuracy translations with iterative refinement, deep contextual understanding, and scalable processing.**

---

## ⚡ **Performance Optimization Tips**
- **Enable CUDA:** If using a compatible NVIDIA GPU, install `torch` with CUDA support (`pip install torch --extra-index-url https://download.pytorch.org/whl/cu118`).
- **Use SSD/NVMe:** Reduces data load time for models.
- **Run in Multi-Process Mode:** Use `threadpoolctl` for efficient parallel execution.
- **Use Adaptive Batching:** Allows efficient handling of large documents without excessive memory consumption.

---

## 📌 Conclusion
- **ILTE-ALT** 🟢 runs on most basic setups with dictionary-based translation.
- **ILTE-ZS** 🔵 is optimized for **batch processing and hybrid translation techniques**.
- **ILTE-ADV** 🟠 requires **AI-powered hardware with semantic understanding**.
- **ILTE-ATI** 🔴 is the most demanding but provides **the highest translation accuracy through iterative refinement and attention mechanisms**.

Choose the right version based on your **hardware capability** and **translation needs**! 🚀

