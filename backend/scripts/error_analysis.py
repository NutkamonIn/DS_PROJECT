import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedTokenizerFast
from tqdm import tqdm
import os
import sys
from typing import List, Dict, Any, Tuple, Union

# ==========================================
# 1. ตั้งค่า Path
# ==========================================
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR: str = os.path.join(BASE_DIR, "models")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
OUTPUT_FILE: str = os.path.join(BASE_DIR, "error_analysis_report.csv")

# Setup import path for utils
sys.path.append(BASE_DIR)
from utils.text_processing import preprocess_text

# ==========================================
# 2. โหลดโมเดล (ครบ 4 โมเดล)
# ==========================================
print("Loading Models for Error Analysis...")
device: str = "cpu"

models = {}
try:
    # 1. BERT
    path = os.path.join(MODELS_DIR, "my_thai_news_model")
    models['bert'] = {
        'tokenizer': AutoTokenizer.from_pretrained(path),
        'model': AutoModelForSequenceClassification.from_pretrained(path).to(device),
        'name': 'WangchanBERTa'
    }
    # 2. XLMR
    path = os.path.join(MODELS_DIR, "xlm_roberta_thai_news")
    if os.path.exists(path):
        models['xlmr'] = {
            'tokenizer': AutoTokenizer.from_pretrained(path),
            'model': AutoModelForSequenceClassification.from_pretrained(path).to(device),
            'name': 'XLM-RoBERTa'
        }
    # 3. LogReg & Vectorizer
    lp, vp = os.path.join(MODELS_DIR, 'logreg_model.pkl'), os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
    models['logreg'] = joblib.load(lp)
    models['vec'] = joblib.load(vp)
    
    # 4. Random Forest
    path = os.path.join(MODELS_DIR, 'randomforest_model.pkl')
    if os.path.exists(path):
        models['rf'] = joblib.load(path)

    print("Models Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Models: {e}")
    exit()

id2label: Dict[int, str] = {0: "World", 1: "Business", 2: "SciTech"}
label2id: Dict[str, int] = {"World": 0, "Business": 1, "SciTech": 2}

# ==========================================
# 3. โหลดข้อมูลทดสอบ
# ==========================================
TEST_FILE: str = os.path.join(DATA_DIR, "11.agnews_thai_test_hard.csv") 
df: pd.DataFrame = pd.read_csv(TEST_FILE)
TEXT_COL = 'body' # หรือใช้รวม headline + body
LABEL_COL = 'topic'

# ==========================================
# 4. ฟังก์ชันทำนาย
# ==========================================
def predict_dl(model_dict, text):
    inputs = model_dict['tokenizer'](text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model_dict['model'](**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].tolist()
        pred_id = int(probs.index(max(probs)))
    return pred_id, max(probs)

def predict_ml(model, vectorizer, text):
    v = vectorizer.transform([text])
    probs = model.predict_proba(v)[0].tolist()
    pred_id = int(probs.index(max(probs)))
    return pred_id, max(probs)

# ==========================================
# 5. รันการวิเคราะห์
# ==========================================
error_records = []

print("Running deep analysis across all models...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    text_raw = f"{row['headline']} {row['body']}"
    text = preprocess_text(text_raw)
    true_label = row[LABEL_COL]
    true_id = label2id.get(true_label, -1)
    if true_id == -1: continue

    # ผลทำนายแต่ละตัว
    p_bert, c_bert = predict_dl(models['bert'], text)
    p_log, c_log = predict_ml(models['logreg'], models['vec'], text)
    
    p_xlmr, c_xlmr = (-1, 0)
    if 'xlmr' in models: p_xlmr, c_xlmr = predict_dl(models['xlmr'], text)
    
    p_rf, c_rf = (-1, 0)
    if 'rf' in models: p_rf, c_rf = predict_ml(models['rf'], models['vec'], text)

    # เก็บกรณีที่ "โมเดลใดโมเดลหนึ่งทายผิด" 
    # โดยเฉพาะ XLM-R ที่น่าจะผิดเยอะ จะได้มีตัวอย่างไปเขียนรายงาน
    if p_bert != true_id or p_log != true_id or p_xlmr != true_id or p_rf != true_id:
        error_records.append({
            "index": idx,
            "text": text_raw,
            "true_label": true_label,
            "pred_wangchan": id2label[p_bert],
            "conf_wangchan": round(c_bert, 4),
            "pred_xlmr": id2label.get(p_xlmr, "N/A"),
            "conf_xlmr": round(c_xlmr, 4),
            "pred_logreg": id2label[p_log],
            "pred_rf": id2label.get(p_rf, "N/A"),
            "error_category": "", # เว้นว่างให้เติม
            "analysis_note": ""   # เว้นว่างให้เติม
        })

# ==========================================
# 6. บันทึกผล
# ==========================================
error_df = pd.DataFrame(error_records)
error_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print(f"\nAnalysis Complete! Found {len(error_df)} potential error cases for report.")
print(f"Saved to: {OUTPUT_FILE}")
print("\nTip: Look at the 'pred_xlmr' column, it will contain many errors to analyze!")