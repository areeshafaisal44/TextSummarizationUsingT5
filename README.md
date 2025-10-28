# 🧠 T5 Text Summarization — Fine-Tuning with Hugging Face Transformers

This project fine-tunes a **T5 (Text-To-Text Transfer Transformer)** model for **abstractive text summarization** using the **CNN/DailyMail news dataset** from Kaggle.  
The goal is to automatically generate concise and coherent summaries from long news articles.

---

## 📘 Overview

Abstractive summarization involves **generating new sentences** that capture the main ideas of the text — similar to how a human writes a summary.  
We fine-tune **`t5-small`** or **`t5-base`** using **Hugging Face Transformers** and evaluate model performance using **ROUGE metrics**.

The complete pipeline includes:
- ✅ Data preprocessing (text + summary)
- ✅ Fine-tuning a pre-trained T5 model
- ✅ Evaluation using ROUGE scores
- ✅ Streamlit web app for interactive summarization
- ✅ GitHub integration for reproducibility

---

## 📂 Dataset

**Dataset used:** [Newspaper Text Summarization (CNN/DailyMail)](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)

Each data sample contains:
- **`article`** → full-length news article  
- **`highlights`** → target summary  

Download directly in Colab:
```python
!kaggle datasets download -d gowrishankarp/newspaper-text-summarization-cnn-dailymail
!unzip newspaper-text-summarization-cnn-dailymail.zip -d data/

⚙️ Model & Tokenizer

We use the T5 architecture available from Hugging Face:

  Model: t5-small or t5-base

Tokenizer: T5Tokenizer

  Both model and tokenizer are loaded from:

  from transformers import T5ForConditionalGeneration, T5Tokenizer

🚀 Fine-Tuning Pipeline
1️⃣ Preprocessing
Clean text and summary pairs
Add prefix "summarize: " to each article (T5 format)
Tokenize using the T5Tokenizer

2️⃣ Training
Fine-tune the model with the Seq2SeqTrainer API
Use ROUGE for evaluation
Save the best model in /content/t5_cnn_summary

3️⃣ Evaluation
Generate summaries on the test set
Evaluate with ROUGE-1, ROUGE-2, and ROUGE-L metrics

📁 Directory Structure
T5-Text-Summarization/
│
├── data/                        # Dataset folder
├── app.py                       # Streamlit app for summarization
├── train_t5.ipynb               # Colab notebook for training
├── t5_cnn_summary/              # Saved fine-tuned model
├── requirements.txt             # Dependencies list
└── README.md                    # Project documentation
