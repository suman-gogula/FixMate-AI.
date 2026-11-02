# FixMate AI - Smart Maintenance Ticket Classifier (NLP)

Type a maintenance complaint in plain English. FixMate AI predicts the **category**, **urgency**,
**department** and **recommended action**. Runs fully offline - no paid APIs, no LLM APIs.

## Project structure and purpose of each file

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface: sidebar, dashboard, prediction page, history, analytics, model page |
| `model.py` | Trains TF-IDF + Logistic Regression, calculates accuracy / confusion matrix, runs the full pipeline (`analyze_complaint`) |
| `preprocess.py` | NLP pipeline: lowercase, punctuation removal, tokenization, stop-word removal, stemming |
| `rules.py` | Rule-based urgency detection, department lookup, recommended action (NOT machine learning) |
| `dataset.csv` | 211 labelled complaints (6 categories) used to train and test the model |
| `requirements.txt` | Python libraries to install |
| `assets/logo.png` | Logo used in the app |
| `history.csv` | Created automatically when you predict complaints (your history) |

## Setup (Windows / macOS / Linux)

```bash
cd FixMate-AI
python -m venv venv

# activate the virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"   # one-time, optional (app has a built-in backup list)

streamlit run app.py
```
The browser opens at http://localhost:8501.

Quick check without the UI: `python model.py` prints accuracy and tests both demo examples.

## Demo steps for class
1. Open **Dashboard** -> click **Load demo complaints** (optional, fills the charts using the real model).
2. Go to **Predict Complaint** -> click **Sample 1** -> **Predict Complaint**.
3. Click **Sample 2** -> **Predict Complaint**.
4. Show **Complaint History** (download CSV), **Analytics**, **Model Performance**.

## Common errors and solutions

| Error | Solution |
|---|---|
| `'streamlit' is not recognized` | Activate the virtual environment, then `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'nltk'` (or sklearn) | `pip install -r requirements.txt` inside the activated venv |
| `LookupError: Resource stopwords not found` | `python -c "import nltk; nltk.download('stopwords')"` (the app also has a backup list) |
| Running scripts is disabled (PowerShell) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` or use Command Prompt |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| `FileNotFoundError: dataset.csv` | Run the command from inside the `FixMate-AI` folder |
| Old results after editing dataset | Press `C` in the app -> "Clear cache", or restart Streamlit |
