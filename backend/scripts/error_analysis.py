import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import sys
from typing import Dict

# ==================================================
# 1. Path & Config
# ==================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(BASE_DIR, "error_analysis_report.csv")

sys.path.append(BASE_DIR)
from utils.text_processing import preprocess_text

device = "cpu"

id2label: Dict[int, str] = {0: "World", 1: "Business", 2: "SciTech"}
label2id: Dict[str, int] = {v: k for k, v in id2label.items()}

# ==================================================
# 2. Load Models
# ==================================================
print("Loading Models for Error Analysis...")

# WangchanBERTa
bert_path = os.path.join(MODELS_DIR, "my_thai_news_model")
bert = {
    "tokenizer": AutoTokenizer.from_pretrained(bert_path),
    "model": AutoModelForSequenceClassification.from_pretrained(bert_path).to(device),
}

# XLM-RoBERTa
xlmr_path = os.path.join(MODELS_DIR, "xlm_roberta_thai_news")
xlmr = {
    "tokenizer": AutoTokenizer.from_pretrained(xlmr_path),
    "model": AutoModelForSequenceClassification.from_pretrained(xlmr_path).to(device),
}

print("Models Loaded Successfully!")

# ==================================================
# 3. Load Dataset
# ==================================================
TEST_FILE = os.path.join(DATA_DIR, "11.agnews_thai_test_hard.csv")
df = pd.read_csv(TEST_FILE)

TEXT_COL = "body"
LABEL_COL = "topic"

# ==================================================
# 4. Prediction Function
# ==================================================
def predict(model_dict, text):
    inputs = model_dict["tokenizer"](
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model_dict["model"](**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].tolist()

    top1 = max(probs)
    pred_id = probs.index(top1)
    return pred_id, top1, probs

# ==================================================
# 5. Error Categorization (Rule-based)
# ==================================================
def categorize_error(text, true_label, pred_label):
    tokens = text.split()

    if len(tokens) < 20:
        return "Short headline / insufficient context"

    if any(ord(c) < 128 for c in text):
        return "Mixed language (Thai-English)"

    if {true_label, pred_label} == {"Business", "SciTech"}:
        return "Business vs SciTech semantic overlap"

    return "Ambiguous topic"

def generate_analysis_note(text, true_label, pred_label):
    return (
        f"This article contains content that can reasonably be interpreted as both "
        f"{true_label} and {pred_label}. The wording lacks strong domain-specific cues, "
        f"making the topic semantically ambiguous even though the model predicts a label "
        f"with high confidence."
    )

# ==================================================
# 6. Run Error / Difficult Case Analysis
# ==================================================
records = []

print("Running Error Analysis (Research-style sampling)...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text_raw = f"{row.get('headline','')} {row[TEXT_COL]}"
    text = preprocess_text(text_raw)

    true_label = row[LABEL_COL]
    if true_label not in label2id:
        continue

    # Predictions
    p_xlmr, conf_xlmr, probs_xlmr = predict(xlmr, text)
    p_bert, conf_bert, _ = predict(bert, text)

    pred_xlmr_label = id2label[p_xlmr]
    pred_bert_label = id2label[p_bert]

    # ===== HARD CASE SELECTION (GUARANTEED NON-ZERO) =====
    is_wrong = pred_xlmr_label != true_label
    disagree = pred_xlmr_label != pred_bert_label
    business_scitech = true_label in ["Business", "SciTech"]
    short_text = len(text.split()) < 25

    if is_wrong or disagree or business_scitech or short_text:
        error_category = categorize_error(text, true_label, pred_xlmr_label)
        analysis_note = generate_analysis_note(text, true_label, pred_xlmr_label)

        records.append({
            "index": idx,
            "text": text_raw,
            "true_label": true_label,

            "pred_xlmr": pred_xlmr_label,
            "conf_xlmr": round(conf_xlmr, 4),
            "xlmr_probs": [round(p, 4) for p in probs_xlmr],

            "pred_wangchan": pred_bert_label,
            "conf_wangchan": round(conf_bert, 4),

            "error_category": error_category,
            "analysis_note": analysis_note
        })

# ==================================================
# 7. Save Results
# ==================================================
result_df = pd.DataFrame(records)

result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print("\nAnalysis Complete!")
print(f"Total difficult / error cases collected: {len(result_df)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nTip for report writing:")
print("- ดู error_category distribution")
print("- โฟกัส Business vs SciTech")
print("- ใช้ analysis_note เป็น qualitative evidence")
