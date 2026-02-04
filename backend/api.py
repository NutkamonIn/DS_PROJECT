from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
import json
import time
import sys
from typing import Any, List, Dict

# Setup import path for utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.text_processing import preprocess_text

app = FastAPI(title="Thai News Classification API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
BENCHMARK_FILE = os.path.join(MODELS_DIR, "benchmark_metrics.json")

# --- Load Static Data ---
STATIC_BENCHMARK_DATA = []
try:
    with open(BENCHMARK_FILE, 'r') as f:
        STATIC_BENCHMARK_DATA = json.load(f)
except:
    STATIC_BENCHMARK_DATA = [
        {"name": "WangchanBERTa", "accuracy": 0, "f1": 0, "time": 0},
        {"name": "XLM-RoBERTa", "accuracy": 0, "f1": 0, "time": 0},
        {"name": "Logistic Reg", "accuracy": 0, "f1": 0, "time": 0},
        {"name": "Random Forest", "accuracy": 0, "f1": 0, "time": 0}
    ]

# --- Load Models ---
print("Loading Models...")
device = "cpu"
models = {}

def load_all_models():
    # 1. WangchanBERTa
    path = os.path.join(MODELS_DIR, "my_thai_news_model")
    if os.path.exists(path):
        models['bert'] = {
            'tokenizer': AutoTokenizer.from_pretrained(path),
            'model': AutoModelForSequenceClassification.from_pretrained(path).to(device)
        }
    
    # 2. XLM-R
    path = os.path.join(MODELS_DIR, "xlm_roberta_thai_news")
    if os.path.exists(path):
        models['xlmr'] = {
            'tokenizer': AutoTokenizer.from_pretrained(path),
            'model': AutoModelForSequenceClassification.from_pretrained(path).to(device)
        }

    # 3. Logistic Reg & Vectorizer
    lp, vp = os.path.join(MODELS_DIR, 'logreg_model.pkl'), os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    if os.path.exists(lp) and os.path.exists(vp):
        models['logreg'] = joblib.load(lp)
        models['vec'] = joblib.load(vp)

    # 4. Random Forest
    path = os.path.join(MODELS_DIR, 'randomforest_model.pkl')
    if os.path.exists(path):
        models['rf'] = joblib.load(path)

load_all_models()
labels = ['World', 'Business', 'SciTech']

class NewsInput(BaseModel):
    text: str

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/model/info")
def model_info():
    return {
        "models": [
            {"name": "WangchanBERTa", "type": "Deep Learning", "version": "1.0.0"},
            {"name": "XLM-RoBERTa", "type": "Deep Learning", "version": "1.0.0"},
            {"name": "Logistic Regression", "type": "Machine Learning", "version": "1.0.0"},
            {"name": "Random Forest", "type": "Machine Learning", "version": "1.0.0"}
        ]
    }

@app.post("/compare")
async def compare(input_data: NewsInput):
    text = preprocess_text(input_data.text)
    predictions = []
    live_latencies = {}

    # --- 1. BERT ---
    if 'bert' in models:
        s = time.perf_counter()
        inputs = models['bert']['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = models['bert']['model'](**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
        lat = round((time.perf_counter() - s) * 1000, 2)
        live_latencies["WangchanBERTa"] = lat
        predictions.append({"model": "WangchanBERTa", "top_class": labels[probs.index(max(probs))], "confidence": max(probs), "latency_ms": lat, "scores": probs})

    # --- 2. XLMR ---
    if 'xlmr' in models:
        s = time.perf_counter()
        inputs = models['xlmr']['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = models['xlmr']['model'](**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
        lat = round((time.perf_counter() - s) * 1000, 2)
        live_latencies["XLM-RoBERTa"] = lat
        predictions.append({"model": "XLM-RoBERTa", "top_class": labels[probs.index(max(probs))], "confidence": max(probs), "latency_ms": lat, "scores": probs})

    # --- 3. LogReg ---
    if 'logreg' in models and 'vec' in models:
        s = time.perf_counter()
        v = models['vec'].transform([text])
        probs = models['logreg'].predict_proba(v)[0].tolist()
        lat = round((time.perf_counter() - s) * 1000, 2)
        live_latencies["Logistic Reg"] = lat
        predictions.append({"model": "Logistic Regression", "top_class": labels[probs.index(max(probs))], "confidence": max(probs), "latency_ms": lat, "scores": probs})

    # --- 4. RF ---
    if 'rf' in models and 'vec' in models:
        s = time.perf_counter()
        v = models['vec'].transform([text])
        probs = models['rf'].predict_proba(v)[0].tolist()
        lat = round((time.perf_counter() - s) * 1000, 2)
        live_latencies["Random Forest"] = lat
        predictions.append({"model": "Random Forest", "top_class": labels[probs.index(max(probs))], "confidence": max(probs), "latency_ms": lat, "scores": probs})

    # Sync live latencies into benchmarks
    benchmarks = []
    for item in STATIC_BENCHMARK_DATA:
        new_item = item.copy()
        # Mapping names to match benchmarks
        m_name = item['name']
        if m_name == "Logistic Reg": lat_key = "Logistic Reg"
        else: lat_key = m_name
        new_item['time'] = live_latencies.get(lat_key, 0)
        benchmarks.append(new_item)

    return {"predictions": predictions, "benchmarks": benchmarks}

@app.get("/evaluation/image/{model_type}")
def get_image(model_type: str):
    file_map = {"bert": "cm_wangchanberta.png", "xlmr": "cm_xlm-roberta.png", "logreg": "cm_logistic_regression.png", "rf": "cm_random_forest.png"}
    path = os.path.join(MODELS_DIR, file_map.get(model_type, ""))
    return FileResponse(path) if os.path.exists(path) else {"error": "not found"}