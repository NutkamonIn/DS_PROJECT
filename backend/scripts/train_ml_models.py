import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import sys

# Setup import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processing import preprocess_text

# ==========================================
# 1. Setup Path & Config
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

# สร้าง folder models ถ้ายังไม่มี
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"📂 Data Directory: {DATA_DIR}")
print(f"📂 Models Directory: {MODELS_DIR}")

# ==========================================
# 2. Load & Prepare Data
# ==========================================
print("⏳ Loading Dataset...")
try:
    # ใช้ไฟล์เดิมที่มีอยู่
    df = pd.read_csv(os.path.join(DATA_DIR, '11.agnews_thai_test_hard.csv'))
    
    # รวม Headline + Body เพื่อให้โมเดลได้ข้อมูลครบถ้วนที่สุด
    df['text'] = (df['headline'] + " " + df['body']).apply(preprocess_text)
    
    # Map Label เป็นตัวเลข (ตรวจสอบชื่อคอลัมน์จากไฟล์จริง)
    # สมมติ label คือ 'topic' และค่าคือ World, Business, SciTech
    if 'topic' in df.columns:
        label_map = {'World': 0, 'Business': 1, 'SciTech': 2}
        df['label'] = df['topic'].map(label_map)
        # กรองแถวที่ map ไม่ได้ออก (ถ้ามี)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
    else:
        print("❌ Error: Column 'topic' not found in dataset.")
        exit()

    print(f"✅ Loaded {len(df)} rows.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ==========================================
# 3. Text Representation (TF-IDF)
# ==========================================
print("⏳ Vectorizing Text (TF-IDF)...")
# Note: ใช้ word-level ตามโจทย์ (analyzer='word')
# ปรับ max_features เพื่อจำกัดขนาด vocabulary ไม่ให้ใหญ่เกินไป
tfidf = TfidfVectorizer(analyzer='word', max_features=5000, ngram_range=(1, 2)) 

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Save Vectorizer (สำคัญ! ต้องใช้ตอน Deploy)
joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
print("✅ Saved TF-IDF Vectorizer")

# ==========================================
# 4. Train Models
# ==========================================

# --- Model A: Logistic Regression (Baseline ตามโจทย์) ---
print("\n🤖 Training Logistic Regression (Baseline)...")
# class_weight='balanced' ตามโจทย์กำหนด
model_logreg = LogisticRegression(
    class_weight='balanced', 
    max_iter=1000, 
    random_state=42,
    solver='lbfgs'
)
start = time.time()
model_logreg.fit(X_train_vec, y_train)
train_time = time.time() - start

# Evaluate
y_pred_logreg = model_logreg.predict(X_test_vec)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"✅ LogReg Accuracy: {acc_logreg:.4f} (Time: {train_time:.2f}s)")
print(classification_report(y_test, y_pred_logreg, target_names=label_map.keys()))

# Save Model
joblib.dump(model_logreg, os.path.join(MODELS_DIR, 'logreg_model.pkl'))


# --- Model B: Random Forest (โมเดลทางเลือก) ---
print("\n🌲 Training Random Forest (Alternative)...")
# Random Forest มักทนทานต่อ noise และไม่ต้องปรับจูนเยอะ
model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1 # ใช้ทุก CPU core
)
start = time.time()
model_rf.fit(X_train_vec, y_train)
train_time = time.time() - start

# Evaluate
y_pred_rf = model_rf.predict(X_test_vec)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"✅ Random Forest Accuracy: {acc_rf:.4f} (Time: {train_time:.2f}s)")
print(classification_report(y_test, y_pred_rf, target_names=label_map.keys()))

# Save Model (ถ้าอยากใช้ตัวนี้แทน ให้แก้ชื่อไฟล์ตอนโหลดใน api.py)
joblib.dump(model_rf, os.path.join(MODELS_DIR, 'randomforest_model.pkl'))

print("\n🎉 All ML Training Complete!")
