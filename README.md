# การจำแนกหมวดหมู่ข่าวภาษาไทย: การเปรียบเทียบ Deep Learning และ Machine Learning

โปรเจคนี้เป็นแอปพลิเคชัน Full-Stack AI สำหรับการเปรียบเทียบและประเมินประสิทธิภาพของโมเดลการประมวลผลภาษาธรรมชาติ (NLP) หลายรูปแบบในการจำแนกหมวดหมู่ข่าวภาษาไทย พัฒนาขึ้นเพื่อเป็นส่วนหนึ่งของงานมอบหมาย Unified MLDS Deployment

---

## 1. โครงสร้างระบบ (System Architecture)

- **Frontend:** Next.js 15+, Tailwind CSS, Recharts (สำหรับการแสดงผลกราฟ)
- **Backend:** FastAPI (Python 3.9+), PyTorch, Scikit-learn
- **โมเดลที่ใช้:** 
  - **Deep Learning:** WangchanBERTa, XLM-RoBERTa
  - **Machine Learning (Baseline):** Logistic Regression, Random Forest

---

## 2. การติดตั้งและตั้งค่า (Installation & Setup)

### การตั้งค่า Backend
1. เข้าไปที่โฟลเดอร์ backend: `cd backend`
2. สร้างและใช้งาน Virtual Environment:
   - macOS/Linux: `python3 -m venv venv && source venv/bin/activate`
   - Windows: `python3 -m venv venv && venv\Scripts\activate`
3. ติดตั้ง Library ที่จำเป็น: `pip install -r requirements.txt`

### การตั้งค่า Frontend
1. เข้าไปที่โฟลเดอร์ frontend: `cd frontend`
2. ติดตั้ง Dependencies: `npm install`

---

## 3. วิธีการใช้งาน: สคริปต์ต่างๆ (Backend)

สคริปต์ทั้งหมดต้องรันจากโฟลเดอร์ `backend` และควรเปิดใช้งาน Virtual Environment ก่อนรัน

### การเทรนโมเดล (Training)
- **เทรนโมเดล Machine Learning:** `python scripts/train_ml_models.py`
  (สร้างโมเดล Logistic Regression และ Random Forest โดยใช้ TF-IDF)
- **เทรนโมเดล WangchanBERTa:** `python scripts/train_wangchan.py`
- **เทรนโมเดล XLM-RoBERTa:** `python scripts/train_xlmr.py`

### การวัดผลและวิเคราะห์ (Evaluation)
- **รัน Benchmark ทั้งหมด:** `python3\ scripts/benchmark_all.py`
  (ประเมินผลโมเดลทั้ง 4 ตัวบนชุดข้อมูลทดสอบ, สร้างไฟล์ `benchmark_metrics.json` และรูปภาพ Confusion Matrix)
- **รันการวิเคราะห์ข้อผิดพลาด:** `python scripts/error_analysis.py`
  (ดึงเคสที่โมเดลทายผิดออกมาไว้ในไฟล์ `error_analysis_report.csv` เพื่อนำไปเขียนรายงาน)

---

## 4. คู่มือการวิเคราะห์ข้อผิดพลาด (Error Analysis Guide)

สำหรับการจัดทำรายงานในขั้นตอนที่ 5 ให้ดำเนินการดังนี้:

1. รันสคริปต์: `python scripts/error_analysis.py`
2. เปิดไฟล์ `error_analysis_report.csv` ด้วย Excel หรือ Google Sheets
3. เลือกตัวอย่างที่โมเดลทายผิดมา 5-10 ตัวอย่าง และระบุประเภทข้อผิดพลาดในคอลัมน์ `error_category` ดังนี้:
   - **Typo / Noise:** ข้อความมีคำผิดหรือสัญลักษณ์ที่ไม่มีความหมายเยอะเกินไป
   - **Mixed Signals:** ข้อความมีความกำกวม มีคำสำคัญของหลายหมวดหมู่ปนกัน
   - **Negation / Sarcasm:** มีการใช้ประโยคปฏิเสธหรือการประชดประชัน
   - **Domain Shift:** มีการใช้ศัพท์สแลงหรือศัพท์เฉพาะทางที่โมเดลไม่เคยเห็นตอนเทรน

---

## 5. วิธีการใช้งานเว็บแอปพลิเคชัน (Web Application)

### การเปิดใช้งานเซิร์ฟเวอร์
1. **เริ่ม Backend:** `uvicorn api:app --reload` (รันที่ http://localhost:8000)
2. **เริ่ม Frontend:** `npm run dev` (รันที่ http://localhost:3000)

### ฟีเจอร์หลักบนหน้าเว็บ
- **หน้าหลัก (AI Model Arena):** กรอกข้อความข่าวภาษาไทย หรือเลือกตัวอย่าง เพื่อดูผลทำนายจาก 4 โมเดลพร้อมกัน รวมถึงค่าความมั่นใจ (Confidence) และเวลาที่ใช้ (Latency)
- **หน้าเปรียบเทียบโมเดล (Model Comparison):** ตารางเปรียบเทียบสถาปัตยกรรม ข้อดี ข้อเสีย และข้อจำกัดของโมเดลแต่ละประเภท
- **หน้ากระดานประเมินผล (Evaluation Dashboard):** แสดงผลรูปภาพ Confusion Matrix ของทุกโมเดลเพื่อวิเคราะห์ความแม่นยำรายคลาส

---

## 6. ขั้นตอนการเตรียมข้อมูล (Preprocessing)

ระบบใช้มาตรฐานการเตรียมข้อมูลเดียวกันทั้งตอนเทรนและใช้งานจริง (อยู่ใน `backend/utils/text_processing.py`):
1. **การจัดการช่องว่าง (Whitespace Normalization):** ลดช่องว่างซ้ำซ้อนให้เหลือช่องเดียว
2. **การแปลงเป็นตัวพิมพ์เล็ก (Lowercase):** สำหรับตัวอักษรภาษาอังกฤษ
3. **การตัดช่องว่าง (Trimming):** ตัดช่องว่างหัวและท้ายข้อความ
4. **ข้อควรระวัง:** ไม่มีการลบ Emoji หรือ Slang เพื่อทดสอบความทนทานของโมเดลตามที่โจทย์กำหนด

---

## 7. โครงสร้างไฟล์ในโปรเจค (Project Structure)

โครงสร้างไฟล์ถูกจัดแบ่งออกเป็นส่วนของ Backend และ Frontend อย่างชัดเจนตามสถาปัตยกรรมแบบแยกส่วน (Decoupled Architecture)

```text
DS_PROJECT/
├── backend/                 # ส่วนประมวลผลและให้บริการ API (FastAPI)
│   ├── data/                # ชุดข้อมูล (Dataset) สำหรับการเทรนและทดสอบ
│   ├── models/              # ไฟล์ Artifacts ของโมเดล และรูปภาพ Confusion Matrix
│   ├── scripts/             # ชุดสคริปต์สำหรับการจัดการวงจรชีวิตของโมเดล (ML Pipeline)
│   │   ├── train_*.py       # สคริปต์สำหรับการเทรนโมเดลแต่ละประเภท
│   │   ├── benchmark_all.py # สคริปต์สำหรับการประเมินผลรวม
│   │   └── error_analysis.py # สคริปต์สำหรับวิเคราะห์ข้อผิดพลาด
│   ├── utils/               # โมดูลเสริมสำหรับการจัดการข้อความ
│   │   └── text_processing.py # มาตรฐานการทำ Preprocessing กลางของระบบ
│   ├── api.py               # จุดเชื่อมต่อหลักของบริการ API
│   └── requirements.txt     # รายการ Library สำหรับการรันระบบ Backend
├── frontend/                # ส่วนหน้าจอการใช้งาน (Next.js 15)
│   ├── app/                 # โครงสร้างหน้าเว็บหลัก (App Router)
│   │   ├── comparison/      # หน้าวิเคราะห์เปรียบเทียบเชิงทฤษฎี
│   │   ├── evaluation/      # หน้าแสดงผลลัพธ์ Confusion Matrix
│   │   └── page.tsx         # หน้าหลักสำหรับการทำนายผลแบบ Real-time
│   ├── tailwind.config.ts   # การตั้งค่าการออกแบบและสไตล์ (Design System)
│   └── package.json         # การตั้งค่า Dependencies ของระบบ Frontend
└── docs/                    # เอกสารอ้างอิงและคู่มือโครงการ
```

### รายละเอียดส่วนสำคัญ

#### **Backend (`/backend`)**
*   **`api.py`**: ทำหน้าที่เป็น Gateway หลักที่โหลดโมเดลทั้ง 4 ประเภทเข้าสู่หน่วยความจำ และให้บริการ Endpoint สำหรับการทำนายผลพร้อมคำนวณ Latency
*   **`scripts/`**: ศูนย์รวมคำสั่งที่ครอบคลุมตั้งแต่การเทรน (Training) ไปจนถึงการวิเคราะห์เชิงลึก (Error Analysis) ช่วยให้การบริหารจัดการโมเดลทำได้ง่ายและเป็นระบบ
*   **`utils/text_processing.py`**: ควบคุมมาตรฐานการเตรียมข้อมูลให้เป็นรูปแบบเดียวกันทั้งระบบ เพื่อป้องกันปัญหาการทำนายผิดพลาดจากการจัดรูปแบบข้อมูลที่ต่างกัน

#### **Frontend (`/frontend`)**
*   **`page.tsx`**: หน้าจอหลักที่ออกแบบมาในสไตล์ "Arena" เพื่อให้ผู้ใช้สามารถเปรียบเทียบผลลัพธ์จากโมเดลหลายตัวได้ในหน้าเดียว พร้อมการแสดงผลด้วยกราฟที่เข้าใจง่าย
*   **`evaluation/` & `comparison/`**: ส่วนสนับสนุนข้อมูลเชิงลึกที่ช่วยให้ผู้ใช้เข้าใจถึงประสิทธิภาพที่แท้จริงและข้อจำกัดของโมเดลแต่ละประเภท

---

**ผู้พัฒนา:** กลุ่มน้ำตากามเทพ

**วันที่ปรับปรุงล่าสุด:** กุมภาพันธ์ 2569
