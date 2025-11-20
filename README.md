# LLM Fine-Tuning with DistilBERT

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)

This repository contains an implementation of **Transfer Learning** techniques applied to Large Language Models (LLMs). Specifically, it focuses on fine-tuning a **DistilBERT** model for sentiment analysis.

This project was developed as part of the **ALTeGraD (Advanced Learning for Text and Graph Data)** course, part of the **Master MVA (Mathématiques Vision Apprentissage)** at **École Polytechnique** within the *Institut Polytechnique de Paris*.

## 🧠 Project Overview

The goal of this lab is to leverage **Self-Supervised Learning** representations trained on large corpora to solve downstream tasks with limited data.

The implementation covers:
* **Feature Extraction:** Using `DistilBERT` as a static feature extractor.
* **Fine-Tuning:** Unfreezing the model weights to adapt the entire architecture to the target task.
* **Performance Comparison:** Analyzing the trade-offs between training speed and accuracy (Feature Extraction vs. Fine-Tuning).

## 📂 Structure

The Jupyter Notebook guides through the following steps:

1.  **Data Preparation:** Loading the **IMDb Movie Reviews dataset** and tokenizing utilizing `DistilBertTokenizerFast`.
2.  **Model Loading:** Loading the pretrained `distilbert-base-uncased` weights from Hugging Face.
3.  **Training:**
    * *Feature Extraction mode:* Freezing the Transformer backbone and training only the classifier head.
    * *Fine-Tuning mode:* Optimizing all parameters end-to-end.
4.  **Inference:** Testing the model on unseen reviews to predict sentiment (Positive/Negative).

## 📊 Dataset

* **Dataset:** [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
* **Task:** Binary Sentiment Classification (Positive/Negative).
* **Preprocessing:** Padding/Truncation to 512 tokens.

## 🚀 Results

The project demonstrates that **Fine-Tuning** significantly outperforms simple Feature Extraction, although it is computationally more expensive.

| Method | Accuracy |
| :--- | :--- |
| **Feature Extraction** (Frozen Weights) | ~85% |
| **Fine-Tuning** (End-to-End) | **>90%** |

*(Note: Specific accuracy values depend on the training run)*

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/maxencedbe/llm-pretraining-finetuning.git](https://github.com/maxencedbe/llm-pretraining-finetuning.git)
    cd llm-pretraining-finetuning
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch transformers datasets scikit-learn tqdm
    ```

3.  **Run the notebook:**
    Open `notebooks/LLM_Pretraining_Finetuning.ipynb` to execute the training pipeline.

---
*This lab was originally designed by Prof. Michalis Vazirgiannis, Dr. Hadi Abdine, and Yang Zhang for the APM 53674 course.*