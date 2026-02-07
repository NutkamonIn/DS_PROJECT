import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import sys

# Setup import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processing import preprocess_text

# ==========================================
# 1. Setup Path
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
OUTPUT_MODEL_DIR = os.path.join(MODELS_DIR, "xlm_roberta_thai_news")

# ==========================================
# 2. Config & Data Loading
# ==========================================
# ใช้ XLM-RoBERTa (Multilingual Model ยอดนิยม)
MODEL_NAME = "xlm-roberta-base" 

print(f"⏳ Loading Dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, '11.agnews_thai_test_hard.csv'))
df['text'] = (df['headline'] + " " + df['body']).apply(preprocess_text)
label_map = {'World': 0, 'Business': 1, 'SciTech': 2}
df['label'] = df['topic'].map(label_map).astype(int)

# Split (ใช้แค่ส่วนเล็กๆ เพื่อ demo ให้เร็ว ถ้าใช้งานจริงให้ใช้ข้อมูลทั้งหมด)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"✅ Train size: {len(train_df)}, Eval size: {len(eval_df)}")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# ==========================================
# 3. Tokenization
# ==========================================
print(f"⏳ Loading Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# ==========================================
# 4. Training
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results_xlmr",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, # ลดลงถ้ารันบนเครื่อง local ไม่ไหว
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1, # เก็บแค่ checkpoint เดียวประหยัดที่
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

print("🚀 Starting Training (XLM-RoBERTa)...")
trainer.train()

# ==========================================
# 5. Save Model
# ==========================================
print(f"💾 Saving model to {OUTPUT_MODEL_DIR}...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("✅ Done!")
