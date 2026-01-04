# NEREL_MultiTask
NER + event/relation classification

# Multi-Task Learning for NER and Event Classification

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Educational_Project-green)

This project implements a **Joint Multi-Task Model** based on BERT architecture to solve two Information Extraction tasks simultaneously:
1.  **Named Entity Recognition (NER):** Token-level classification (BIO tagging).
2.  **Event/Relation Classification:** Document-level multi-label classification.

The model is trained on the **NEREL** dataset and features advanced techniques such as **Uncertainty-Weighted Loss** for task balancing and **Dynamic Quantization** for inference optimization.

## 🚀 Key Features

* **Joint Architecture:** A shared BERT encoder with separate heads for token classification (NER) and sequence classification (Events).
* **Uncertainty Weighting:** Implemented learnable loss weights ($\sigma$) to automatically balance the training signal between NER and Event tasks (Homoscedastic Uncertainty Learning).
* **Robust Preprocessing:** Custom tokenizer alignment handling subword fragmentation and `offset_mapping`.
* **Inference Optimization:** Benchmarked original GPU model against a Dynamically Quantized (Int8) CPU model for edge deployment scenarios.

## 🛠️ Project Structure

The project is organized as a comprehensive Jupyter Notebook covering the end-to-end ML pipeline:

1.  **Exploratory Data Analysis (EDA):** Class distribution and dataset statistics.
2.  **Data Pipeline:** Parsing NEREL format, custom `Dataset` class, and `DataCollator`.
3.  **Modeling:** Implementation of `JointModel` with `nn.CrossEntropyLoss` (ignore_index=-100) and `nn.BCEWithLogitsLoss`.
4.  **Training:** Training loop with evaluation on the Dev set.
5.  **Optimization:** Post-training Dynamic Quantization.

## 📊 Performance & Benchmarks

### Quality Metrics (Dev Set)
| Task | Metric | Score |
| :--- | :--- | :--- |
| **NER (Token-level)** | F1-Score (Seqeval) | **~0.80** |
| **Event Classification** | Micro-F1 | **0.75** |
| **Event Classification** | Precision | **0.79** |

### Inference Speedup (Quantization)
To simulate production constraints, I compared the original FP32 model (GPU/CPU) with an Int8 Quantized model (CPU).

| Model Version | Device | Avg Batch Time | Speedup | F1 Score Impact |
| :--- | :--- | :--- | :--- | :--- |
| Original | CPU | 9.64s | 1.0x | 0.81 (Base) |
| **Quantized (Int8)** | **CPU** | **8.45s** | **1.14x** | **0.79 (Minimal drop)** |

> *Note: While the GPU baseline is faster in absolute terms, quantization demonstrates a viable strategy for CPU-only environments with minimal accuracy loss.*

## 🧠 Error Analysis & Conclusion

While the model demonstrates robust extraction of main entities, a qualitative analysis revealed the following:
* **Semantic Ambiguity:** The model occasionally confuses semantically close tags like `PROFESSION` vs `ORGANIZATION` in complex contexts.
* **Boundary Issues:** Long compound entities (e.g., media titles) sometimes suffer from fragmented spans.
* **Class Imbalance:** The event classifier tends to be overconfident in frequent classes (`PARTICIPANT_IN`) while under-predicting rare relations (`HAS_CAUSE`).
