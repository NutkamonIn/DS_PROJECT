import pandas as pd
import joblib
import json
import os
import time
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
import sys

# Setup import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processing import preprocess_text

# Setup Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

print(f"Starting Evaluation in: {BACKEND_DIR}")

# 1. Load Data
df = pd.read_csv(os.path.join(DATA_DIR, '11.agnews_thai_test_hard.csv'))
df['text'] = (df['headline'] + " " + df['body']).apply(preprocess_text)
df['label'] = df['topic'].map({'World': 0, 'Business': 1, 'SciTech': 2})
# ใช้ข้อมูลทั้งหมดในการ Evaluate เพื่อความแม่นยำของรายงาน
X_test = df['text']
y_test = df['label']

labels = ['World', 'Business', 'SciTech']
results = []

def save_confusion_matrix(y_true, y_pred, model_name):
    """ฟังก์ชันสำหรับสร้างและบันทึกภาพ Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    file_path = os.path.join(MODELS_DIR, f'cm_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(file_path)
    plt.close()
    print(f"Saved Confusion Matrix to: {file_path}")

# 2. Test ML Models (Logistic Regression & Random Forest)
print("Evaluating ML Models...")
# Load LogReg
model_logreg = joblib.load(os.path.join(MODELS_DIR, 'logreg_model.pkl'))
vec_logreg = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
y_pred_logreg = model_logreg.predict(vec_logreg.transform(X_test))

save_confusion_matrix(y_test, y_pred_logreg, "Logistic Regression")
results.append({
    "name": "Logistic Reg",
    "accuracy": round(accuracy_score(y_test, y_pred_logreg) * 100, 2),
    "f1": round(f1_score(y_test, y_pred_logreg, average='macro') * 100, 2),
    "time": 0
})

# Load Random Forest
try:
    model_rf = joblib.load(os.path.join(MODELS_DIR, 'randomforest_model.pkl'))
    y_pred_rf = model_rf.predict(vec_logreg.transform(X_test))
    save_confusion_matrix(y_test, y_pred_rf, "Random Forest")
    results.append({
        "name": "Random Forest",
        "accuracy": round(accuracy_score(y_test, y_pred_rf) * 100, 2),
        "f1": round(f1_score(y_test, y_pred_rf, average='macro') * 100, 2),
        "time": 0
    })
except FileNotFoundError:
    print("⚠️ Random Forest model not found. Skipping.")

# 3. Test Deep Learning Models
# WangchanBERTa
print("Evaluating WangchanBERTa...")
try:
    bert_path = os.path.join(MODELS_DIR, "my_thai_news_model")
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = AutoModelForSequenceClassification.from_pretrained(bert_path)
    
    dataset = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))
    tokenized = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512), batched=True)
    trainer = Trainer(model=model)
    preds_bert = trainer.predict(tokenized).predictions.argmax(-1)

    save_confusion_matrix(y_test, preds_bert, "WangchanBERTa")
    results.append({
        "name": "WangchanBERTa",
        "accuracy": round(accuracy_score(y_test, preds_bert) * 100, 2),
        "f1": round(f1_score(y_test, preds_bert, average='macro') * 100, 2),
        "time": 0
    })
except Exception as e:
    print(f"⚠️ Error evaluating WangchanBERTa: {e}")

# XLM-RoBERTa
print("Evaluating XLM-RoBERTa...")
try:
    xlmr_path = os.path.join(MODELS_DIR, "xlm_roberta_thai_news")
    tokenizer_xlmr = AutoTokenizer.from_pretrained(xlmr_path)
    model_xlmr = AutoModelForSequenceClassification.from_pretrained(xlmr_path)
    
    dataset_xlmr = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))
    tokenized_xlmr = dataset_xlmr.map(lambda x: tokenizer_xlmr(x["text"], padding="max_length", truncation=True, max_length=128), batched=True)
    trainer_xlmr = Trainer(model=model_xlmr)
    preds_xlmr = trainer_xlmr.predict(tokenized_xlmr).predictions.argmax(-1)

    save_confusion_matrix(y_test, preds_xlmr, "XLM-RoBERTa")
    results.append({
        "name": "XLM-RoBERTa",
        "accuracy": round(accuracy_score(y_test, preds_xlmr) * 100, 2),
        "f1": round(f1_score(y_test, preds_xlmr, average='macro') * 100, 2),
        "time": 0
    })
except Exception as e:
    print(f"⚠️ Error evaluating XLM-RoBERTa: {e}")


# 4. Save Metrics
output_path = os.path.join(MODELS_DIR, "benchmark_metrics.json")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print("\n--- Evaluation Summary ---")
for r in results:
    print(f"Model: {r['name']}")
    print(f"  - Accuracy: {r['accuracy']}%")
    print(f"  - Macro-F1: {r['f1']}%")

print(f"\nAll metrics saved to: {output_path}")
print("Evaluation Complete!")