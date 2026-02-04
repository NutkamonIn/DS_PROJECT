# Thai News Classification: Deep Learning vs Machine Learning

A comprehensive Full-Stack AI Application designed to compare and evaluate multiple Natural Language Processing (NLP) models for Thai news classification. This project is developed as part of the Unified MLDS Deployment Assignment.

## System Architecture

- **Frontend:** Next.js (React), Tailwind CSS, Recharts
- **Backend:** FastAPI (Python), PyTorch, Scikit-learn
- **Models:** 
  - **WangchanBERTa:** Fine-tuned Thai Transformer model.
  - **XLM-RoBERTa:** Multilingual Transformer model.
  - **Logistic Regression:** TF-IDF based baseline model.
  - **Random Forest:** Ensemble learning baseline model.

---

## 1. Installation & Setup

### Backend Setup
Open a terminal and run the following commands:

```bash
cd backend
python3 -m venv venv
# For macOS/Linux:
source venv/bin/activate
# For Windows:
# venv\Scripts\activate

pip install -r requirements.txt
```

### Frontend Setup
Open a new terminal and run:

```bash
cd frontend
npm install
```

---

## 2. Training & Evaluation Scripts

All scripts must be run from the `backend` directory with the virtual environment activated.

### Training Models
To retrain the models from scratch:

1. **Train Machine Learning Models:**
   ```bash
   python3 scripts/train_ml_models.py
   ```
   This will train and save Logistic Regression and Random Forest models.

2. **Train WangchanBERTa:**
   ```bash
   python3 scripts/train_wangchan.py
   ```

3. **Train XLM-RoBERTa:**
   ```bash
   python3 scripts/train_xlmr.py
   ```

### Performance Benchmark
To generate accuracy metrics (F1, Accuracy) and Confusion Matrix images:

```bash
python3 scripts/benchmark_all.py
```
Outputs are saved in `backend/models/`.

---

## 3. Error Analysis Guide (For Report)

This project includes an automated tool to identify misclassified cases for your report.

### Step 1: Run the Analysis Script
```bash
cd backend
python3 scripts/error_analysis.py
```

### Step 2: Open the Report
The script generates a file named `error_analysis_report.csv` in the backend directory. Open this file using Excel or Google Sheets.

### Step 3: Analyze the Errors
Filter the rows where models made incorrect predictions (e.g., look at the 'pred_xlmr' column). For the assignment report, select 5-10 interesting cases and categorize them into the 'error_category' column using these types:

- **Typo / Noise:** The text contains spelling errors or irrelevant symbols.
- **Mixed Signals:** The text contains conflicting keywords (e.g., both political and economic terms).
- **Negation / Sarcasm:** The text uses double negatives or sarcastic language.
- **Domain Shift:** The text uses specific slang or technical terms not present in the training data.

Use these examples to complete Section 6 (Error Analysis) of your final report.

---

## 4. Web Application Usage

### Starting the Servers
1. **Backend:** `uvicorn api:app --reload` (Runs at http://localhost:8000)
2. **Frontend:** `npm run dev` (Runs at http://localhost:3000)

### Features Guide
- **Home (AI Model Arena):** 
  - Enter text or select an example to see real-time predictions from all 4 models.
  - View confidence scores and processing latency (speed) for each model.
- **Model Comparison:** 
  - Click "Model Comparison" to view a detailed table comparing Architecture, Pros, Cons, and Limitations of each model type.
- **Evaluation Dashboard:** 
  - Click "Evaluation Dashboard" to view the Confusion Matrices for all models to visualize prediction performance.

---

## 5. Preprocessing Pipeline

The project enforces a consistent preprocessing logic across training and inference phases, located in `backend/utils/text_processing.py`:
1. **Whitespace Normalization:** Reduces multiple spaces to a single space.
2. **Lowercasing:** Converts English characters to lowercase.
3. **Trimming:** Removes leading and trailing whitespace.
4. **No Over-cleaning:** Emojis and special characters are preserved to test model robustness against noise.

---
**Developers:** [Your Group Name]
