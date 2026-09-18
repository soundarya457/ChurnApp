# ChurnApp — Setup & Run Guide

## What This App Does
Upload any CSV dataset → auto-detects columns → trains 4 ML models →
shows predictions, risk segments, SHAP explanations on a web dashboard.

---

## Prerequisites

| Tool        | Version   | Download |
|-------------|-----------|----------|
| Python      | 3.10–3.12 | https://python.org/downloads |
| MySQL       | 8.0+      | https://dev.mysql.com/downloads/mysql |
| VS Code     | Any       | https://code.visualstudio.com |

---

## One-Time Setup (do this once)

### 1. Open the folder in VS Code
File → Open Folder → select the ChurnApp folder

### 2. Open the terminal in VS Code
Terminal → New Terminal  (or press Ctrl + `)

### 3. Create a virtual environment
```bash
python -m venv venv
```

### 4. Activate it
```bash
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```
You will see (venv) in your terminal prompt.

### 5. Install all libraries
```bash
pip install -r requirements.txt
```
This takes 3–7 minutes. Wait until it finishes.

### 6. Edit your MySQL credentials
Open  src/config.py  and change these two lines:
```python
"user":     "root",          # your MySQL username
"password": "your_password", # your MySQL password
```

### 7. Create the database in MySQL
Open MySQL Workbench, connect, and run:
```sql
CREATE DATABASE IF NOT EXISTS churn_db;
```

### 8. Create the output tables
In MySQL Workbench:
File → Open SQL Script → select  sql/01_create_tables.sql → click ⚡

---

## Every Time You Want to Use the App

### Step 1 — Activate the virtual environment
```bash
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

### Step 2 — Start the web app
```bash
python app.py
```

### Step 3 — Open in browser
Go to:  http://localhost:5000

---

## How to Use the Dashboard

1. Click **Upload Dataset** in the left sidebar
2. Drag and drop your CSV file (any CSV with a binary churn/target column)
3. The app shows auto-detected columns — verify they look correct
4. Click **Run Full Pipeline** — watch live progress
5. When done, navigate to:
   - **Overview** — KPIs, charts, churn rate
   - **Customer Risk Table** — all predictions, filter by risk
   - **Model Evaluation** — ROC curves, confusion matrices
   - **SHAP Explainability** — what drives each prediction

---

## Dataset Requirements

Your CSV must have:
- At least 50 rows
- At least 3 columns
- One binary column (2 unique values) that represents churn vs not-churn
  Example: "Attrition_Flag" with values "Attrited Customer" / "Existing Customer"
  Example: "churned" with values 1 / 0
  Example: "status" with values "left" / "active"

Everything else is auto-detected — no configuration needed.

---

## Project Structure

```
ChurnApp/
├── app.py                 ← Flask web server (run this)
├── main.py                ← CLI pipeline runner
├── requirements.txt       ← Python libraries
├── SETUP.md               ← This file
├── src/
│   ├── config.py          ← DB credentials (edit this)
│   ├── ingest.py          ← CSV → MySQL (auto-detects schema)
│   ├── preprocess.py      ← Feature engineering + SMOTE
│   ├── train.py           ← 4 ML models
│   ├── evaluate.py        ← Metrics + plots
│   ├── explain.py         ← SHAP + LIME
│   └── predict.py         ← Batch scoring → MySQL
├── sql/
│   ├── 01_create_tables.sql   ← Run once in MySQL Workbench
│   └── 03_useful_queries.sql  ← Useful queries to explore results
├── templates/
│   └── index.html         ← Web dashboard (all 5 pages)
├── data/                  ← Uploaded CSV files saved here
├── outputs/               ← ML plots and model saved here
└── uploads/               ← Temp upload folder
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `No module named 'flask'` | Run: `venv\Scripts\activate` then `pip install -r requirements.txt` |
| `Access denied for user root` | Wrong password in `src/config.py` |
| `No CSV found in data/` | Upload a CSV from the dashboard first |
| `Could not detect a binary target column` | Your CSV needs a column with exactly 2 unique values |
| `mysql not recognized` | Add MySQL bin folder to Windows PATH |
| `Can't connect to MySQL` | Start MySQL from MySQL Workbench or Windows Services |

---

## Run Pipeline from Terminal (optional)

Instead of using the web UI, you can run the pipeline directly:
```bash
# Uses the last uploaded CSV automatically
python main.py

# Point to a specific CSV file
python main.py --csv data/myfile.csv

# Run with Optuna hyperparameter tuning (slower, better results)
python main.py --tune
```
