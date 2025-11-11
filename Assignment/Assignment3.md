# Assignment 3: IMDB Sentiment Classification using Bi-LSTM

## Overview
In this assignment, you will build a **sentiment classification model** using the **IMDB Movie Review Dataset** (50 000 labeled reviews).  
Your goal is to classify each review as **positive** or **negative**.

**Due** on 2025-11-21 11:59 PM.

You must:
- Implement a **Bidirectional LSTM (Bi-LSTM)** model (**no attention**)  
- Use **PyTorch Lightning** for modular training  
- Track experiments with **Weights & Biases (W&B)**  
- Use a **BERT tokenizer** for tokenization, but **train your own `nn.Embedding`**
- Teamwork: everyone needs to contribute to this assignment. I will grade individuals based on the peer evaluation.

---

## 1. Dataset
- Use the **IMDB 50 K dataset** (from `torchtext` or Hugging Face `datasets`).  
- Split: **70 % train / 15 % validation / 15 % test**  
- Fix random seed = **42** for reproducibility.

---

## 2. Data Preprocessing
- Tokenize with **`transformers.BertTokenizer`** (e.g. `'bert-base-uncased'`).  
- Pad or truncate each sequence to a maximum length (e.g. 256 tokens).  
- Include special tokens `[CLS]`, `[SEP]`, `[PAD]`.  
- Convert tokens → IDs → tensors → `DataLoader`s.

> ⚠️ Use **only the BERT tokenizer** — it provides a consistent vocabulary and sub-word segmentation.  
> **Do NOT** load `BertModel` or use any pre-trained encoder.  
> Your model must contain a trainable `nn.Embedding` layer.

---

### Example Code for Tokenizer + Embedding
```python
from transformers import BertTokenizer
import torch
import torch.nn as nn

# 1. Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. Tokenize a sentence
text = "I really enjoyed this movie!"
encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)
input_ids = encoding["input_ids"]          # shape: [1, 256]

# 3. Define your own trainable embedding
vocab_size = tokenizer.vocab_size          # 30522
embed_dim  = 128
embedding  = nn.Embedding(vocab_size, embed_dim)

# 4. Obtain embeddings for tokens
# This is your input data
embedded = embedding(input_ids)            # shape: [1, 256, 128]
```
These embeddings will be fed into your Bi-LSTM network.


## 3. Model Architecture
- **Embedding layer:** trainable, no pre-trained weights  
- **Bidirectional LSTM:** at least one Bi-LSTM layer (dropout optional)  
- **Fully connected layer:** output = 2 classes (positive/negative)  
- **Activation:** Sigmoid or Softmax  
- **No attention or transformer components allowed**

---

## 4. Implementation (PyTorch Lightning)
Use **PyTorch Lightning** to organize your code:
- `LightningModule`: defines the model, optimizer, loss, and training/validation/test steps  
- `LightningDataModule`: handles tokenization, padding, and data loading  
- Use **CrossEntropyLoss** for classification  
- Optimizer: **Adam** or **AdamW**

---

## 5. Experiment Tracking (Weights & Biases)
You must log the following metrics using **Weights & Biases (W&B)**:
- Training and validation loss per epoch  
- Validation accuracy  
- Key hyperparameters (learning rate, batch size, sequence length, dropout rate, etc.)  

Include your **W&B project link** in your report.

---

## 6. Evaluation
You must report:
- **Test accuracy**  
- **Confusion matrix**  
- At least **3 misclassified examples**  
- (Optional) **Precision**, **Recall**, and **F1-score**

---

## 7. Report (Maximum 3 Pages)
Your written report should include:
1. **Model architecture** and chosen hyperparameters  
2. **Training setup** (optimizer, learning rate, batch size, etc.)  
3. **Loss and accuracy curves** from W&B  
4. **Final evaluation results**  
5. **Example misclassifications and error analysis**  
6. **Discussion and conclusion**

---

## 8. Submission Instructions
You must submit a **GitHub repository** that includes:
- Your complete source code (`.py`)  
- A `README.md` file with clear run instructions  
- A PDF report (max 3 pages)  
- A link to your **W&B project**

---

## 9. Grading Rubric (100 Points)

| Category | Description | Points |
|-----------|--------------|--------|
| Implementation correctness | Proper use of PyTorch Lightning, Bi-LSTM, and W&B | 30 |
| Model performance | Reasonable accuracy (> 85% train / > 80% val) | 20 |
| Code quality & organization | Clean, modular, reproducible code | 15 |
| Report clarity & insight | Includes curves, analysis, and meaningful discussion | 25 |
| Reproducibility | Fixed random seed and clear instructions | 10 |


---

## Why BERT Tokenizer but Not BERT Model
`BertTokenizer` is used **only** to convert text into integer IDs using BERT’s standard vocabulary.  
It ensures consistent tokenization and subword handling.  
You **must not** use `BertModel` or any pre-trained transformer encoder.  
Your model’s `nn.Embedding` layer is **randomly initialized and trained from scratch**, ensuring the learning process is independent of pre-trained models.

**Dataset Reference:** [IMDB 50K Dataset](https://ai.stanford.edu/~amaas/data/sentiment/). You can use torchtext to access data by: https://docs.pytorch.org/text/stable/datasets.html. If so, you can use default parameter for tran and test datast split. But, you still need to split the train set as train and val sets.
