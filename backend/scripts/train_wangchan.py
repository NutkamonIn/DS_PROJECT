import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import sys

# Setup import path for utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.text_processing import preprocess_text

# === 1. Setup Path (ตั้งค่าที่อยู่ไฟล์) ===
# หาตำแหน่งโฟลเดอร์ปัจจุบัน แล้วถอยกลับไปหา backend
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
OUTPUT_MODEL_DIR = os.path.join(MODELS_DIR, "my_thai_news_model")

# สร้างโฟลเดอร์เก็บโมเดลถ้ายังไม่มี
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# === 2. Config Device (ตั้งค่าการ์ดจอ) ===
# เช็คว่าเครื่องมี GPU อะไรให้ใช้บ้าง (Mac M1/M2 ใช้ mps)
if torch.cuda.is_available():
    device = "cuda"
    print("🚀 Using GPU (CUDA)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("🍎 Using Mac GPU (MPS) - แรงแน่นอน!")
else:
    device = "cpu"
    print("🐢 Using CPU (Might be slow)")

# === 3. Load & Prepare Data (เตรียมข้อมูล) ===
print(f"📂 Loading Data from: {DATA_DIR}")
csv_path = os.path.join(DATA_DIR, '11.agnews_thai_test_hard.csv')

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("❌ ไม่เจอไฟล์ CSV! กรุณาเช็คว่าไฟล์ 11.agnews_thai_test_hard.csv อยู่ใน folder 'backend/data' แล้วหรือยัง")
    exit()

# รวมหัวข้อกับเนื้อหาข่าวเข้าด้วยกัน และทำ Preprocessing
df['text'] = (df['headline'] + " " + df['body']).apply(preprocess_text)
label_map = {'World': 0, 'Business': 1, 'SciTech': 2}
df['label'] = df['topic'].map(label_map)

# แบ่งข้อมูล Train 80% / Validation 20%
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# แปลงข้อมูลให้เป็น Format ของ Hugging Face Dataset
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))

# === 4. Tokenizer (ตัวตัดคำ) ===
MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
print(f"⬇️ Downloading Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # ตัดคำและแปลงเป็นตัวเลข (Padding ให้เท่ากันที่ 128)
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("⚙️ Tokenizing data...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# === 5. Model Setup (โหลดโมเดล) ===
print("⬇️ Downloading Model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(device) # ส่งโมเดลไปที่ GPU/MPS

# === 6. Training Arguments (ตั้งค่าการเทรน) ===
training_args = TrainingArguments(
    output_dir="./results_temp",    # โฟลเดอร์ชั่วคราว
    num_train_epochs=3,             # เทรน 3 รอบ (ถ้าเครื่องร้อนลดเหลือ 1-2 ได้)
    per_device_train_batch_size=8,  # ขนาด Batch (ถ้า RAM ไม่พอให้ลดเหลือ 4)
    per_device_eval_batch_size=8,
    
    evaluation_strategy="epoch",          
    
    save_strategy="no",             # ไม่ต้องเซฟ Checkpoint ระหว่างทาง (เปลืองที่)
    learning_rate=2e-5,             # อัตราการเรียนรู้
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

# === 7. Start Training (เริ่มเทรน) ===
print("🚀 Start Training... (ขั้นตอนนี้ใช้เวลาสักพัก เตรียมกาแฟรอได้เลย ☕)")
trainer.train()

# === 8. Save Model (บันทึกผลลัพธ์) ===
print(f"💾 Saving model to: {OUTPUT_MODEL_DIR}")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)

print("🎉 Training Complete! WangchanBERTa is ready for action.")