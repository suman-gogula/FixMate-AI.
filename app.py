"""
app.py - Streamlit front-end for FixMate AI
Run with:  streamlit run app.py
"""

import html
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import streamlit as st

from model import analyze_complaint, train_model
from rules import CATEGORIES, DEPARTMENTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
HISTORY_FILE = os.path.join(BASE_DIR, "history.csv")
HISTORY_COLUMNS = ["Complaint ID", "Complaint", "Category", "Priority",
                   "Department", "Action", "Confidence", "Time"]
DEPARTMENT_LIST = list(DEPARTMENTS.values())
URGENCY_ORDER = ["High", "Medium", "Low"]
URGENCY_COLORS = {"High": "#DC2626", "Medium": "#F59E0B", "Low": "#16A34A"}
BLUE = "#1D4ED8"

SAMPLE_1 = "Water is leaking continuously from my bathroom ceiling."
SAMPLE_2 = "The light bulb in my bedroom is not working."

DEMO_COMPLAINTS = [
    SAMPLE_1, SAMPLE_2,
    "The AC is not cooling in the seminar hall.",
    "Exposed wires are hanging near the switch board.",
    "My study chair is broken.",
    "Garbage is not collected from the corridor.",
    "The kitchen tap is dripping all night.",
    "There is a big crack in the wall of my room.",
    "The drain in my bathroom is blocked.",
    "Dust is everywhere in the classroom.",
    "I can smell gas in the kitchen.",
    "The cupboard door is not closing.",
]

PAGES = ["Dashboard", "Predict Complaint", "Complaint History",
         "Analytics", "Model Performance", "How It Works"]

st.set_page_config(page_title="FixMate AI", page_icon="🛠️", layout="wide")

# ---------------------------------------------------------------------------
# Styling (blue + white SaaS look)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.stApp { background: #F3F6FB; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B2A6F 0%, #1D4ED8 100%); }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
.card { background:#FFFFFF; border-radius:14px; padding:18px 20px;
        box-shadow:0 1px 4px rgba(15,23,42,0.08); border:1px solid #E5EAF3; margin-bottom:12px; }
.card h4 { margin:0 0 6px 0; color:#0B2A6F; }
.metric-title { font-size:0.85rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.metric-value { font-size:2.1rem; font-weight:700; line-height:1.2; }
.metric-sub { font-size:0.8rem; color:#94A3B8; }
.badge { display:inline-block; padding:6px 16px; border-radius:999px; color:#fff; font-weight:700; font-size:1rem; }
.label { font-size:0.8rem; color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.value { font-size:1.15rem; color:#0F172A; font-weight:600; }
.banner { background:linear-gradient(90deg,#0B2A6F,#1D4ED8); color:#fff; border-radius:14px; padding:18px 24px; margin-bottom:16px; }
.banner h2 { color:#fff; margin:0; }
.banner p { color:#DBEAFE; margin:2px 0 0 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model + history helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the NLP model...")
def get_model():
    return train_model()


def load_history() -> pd.DataFrame:
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_FILE, index=False)


def add_to_history(result: dict):
    df = st.session_state.history
    row = {
        "Complaint ID": f"FM-{len(df) + 1:04d}",
        "Complaint": result["original"],
        "Category": result["category"],
        "Priority": result["urgency"],
        "Department": result["department"],
        "Action": result["action"],
        "Confidence": round(result["confidence"] * 100, 1),
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state.history = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_history(st.session_state.history)


def clear_history():
    st.session_state.history = pd.DataFrame(columns=HISTORY_COLUMNS)
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    st.session_state.last_result = None


def load_demo():
    bundle = get_model()
    for text in DEMO_COMPLAINTS:
        add_to_history(analyze_complaint(bundle, text))


def set_sample(text: str):
    st.session_state["complaint_text"] = text


def go_to(page: str):
    st.session_state["nav"] = page


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def metric_card(title, value, color=BLUE, sub=""):
    st.markdown(
        f"""<div class="card"><div class="metric-title">{title}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)


def urgency_badge(level):
    icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[level]
    return (f'<span class="badge" style="background:{URGENCY_COLORS[level]}">'
            f'{icon} {level.upper()}</span>')


def info_block(label, value):
    st.markdown(f'<div class="card"><div class="label">{label}</div>'
                f'<div class="value">{html.escape(str(value))}</div></div>',
                unsafe_allow_html=True)


def bar_chart(series: pd.Series, colors, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(series.index, series.values, color=colors)
    ax.bar_label(bars, padding=2)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#0B2A6F")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def category_counts(df):
    return df["Category"].value_counts().reindex(CATEGORIES, fill_value=0)


def urgency_counts(df):
    return df["Priority"].value_counts().reindex(URGENCY_ORDER, fill_value=0)


def department_counts(df):
    return df["Department"].value_counts().reindex(DEPARTMENT_LIST, fill_value=0)


def show_result(r: dict):
    st.markdown("### Prediction Result")
    c1, c2 = st.columns(2)
    with c1:
        info_block("Original complaint", r["original"])
    with c2:
        info_block("Cleaned (preprocessed) complaint", r["clean"] or "(empty after cleaning)")

    with st.expander("See every preprocessing step"):
        s = r["steps"]
        st.write("**1. Lowercase:**", s["lowercase"])
        st.write("**2. Punctuation removed:**", s["no_punctuation"])
        st.write("**3. Tokens:**", s["tokens"])
        st.write("**4. After stop-word removal:**", s["no_stopwords"])
        st.write("**5. After stemming:**", s["stemmed"])

    c1, c2, c3 = st.columns(3)
    with c1:
        info_block("Predicted category (ML model)", r["category"])
        st.progress(min(max(r["confidence"], 0.0), 1.0),
                    text=f"Confidence: {r['confidence'] * 100:.1f}%")
        if r["low_confidence"]:
            st.warning("Low confidence - the complaint may not match the training data well.")
    with c2:
        st.markdown(
            f'<div class="card"><div class="label">Urgency (rule-based)</div>'
            f'{urgency_badge(r["urgency"])}'
            f'<div class="metric-sub" style="margin-top:8px">Rule matched: {html.escape(r["urgency_reason"])}</div></div>',
            unsafe_allow_html=True)
    with c3:
        info_block("Assigned department", r["department"])

    info_block("Recommended action", r["action"])
    if r["safety"]:
        st.warning(f"Safety tip: {r['safety']}")

    with st.expander("Probability of each category"):
        prob = pd.Series(r["probabilities"]).reindex(CATEGORIES) * 100
        st.bar_chart(prob)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard():
    df = st.session_state.history
    c1, c2 = st.columns([1, 9])
    with c1:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=80)
    with c2:
        st.markdown("""<div class="banner"><h2>FixMate AI</h2>
        <p>Smart Maintenance Ticket Classifier using Natural Language Processing</p></div>""",
                    unsafe_allow_html=True)

    st.markdown("""<div class="card"><h4>Welcome 👋</h4>
    Type a maintenance complaint in plain English. FixMate AI cleans the text with an NLP pipeline,
    predicts the <b>category</b> with a machine-learning model, decides the <b>urgency</b> using clear rules,
    and assigns the <b>department</b> and <b>recommended action</b>.</div>""", unsafe_allow_html=True)

    total = len(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total complaints", total, BLUE, "processed so far")
    with c2:
        metric_card("High priority", int((df["Priority"] == "High").sum()), URGENCY_COLORS["High"], "needs immediate action")
    with c3:
        metric_card("Medium priority", int((df["Priority"] == "Medium").sum()), URGENCY_COLORS["Medium"], "fix in 24-48 hours")
    with c4:
        metric_card("Low priority", int((df["Priority"] == "Low").sum()), URGENCY_COLORS["Low"], "routine maintenance")

    st.markdown("#### Complaint category statistics")
    if total == 0:
        st.info("No complaints processed yet. Try the Predict Complaint page, or load demo complaints "
                "(they are run through the real model, not fake numbers).")
    else:
        counts = category_counts(df)
        left, right = st.columns([3, 2])
        with left:
            bar_chart(counts, BLUE, "Complaints by category")
        with right:
            st.dataframe(counts.rename("Complaints").reset_index().rename(columns={"index": "Category"}),
                         hide_index=True)

    b1, b2, b3 = st.columns([2, 2, 3])
    with b1:
        st.button("Predict a complaint", on_click=go_to, args=("Predict Complaint",), type="primary")
    with b2:
        st.button("Load demo complaints", on_click=load_demo)
    with b3:
        st.button("Clear history", on_click=clear_history)


def page_predict():
    bundle = get_model()
    st.markdown("## Predict Complaint")
    st.caption("Enter a complaint or click a sample below.")

    st.session_state.setdefault("complaint_text", "")
    s1, s2 = st.columns(2)
    with s1:
        st.button("Sample 1: Bathroom ceiling leak", on_click=set_sample, args=(SAMPLE_1,))
    with s2:
        st.button("Sample 2: Bedroom light bulb", on_click=set_sample, args=(SAMPLE_2,))

    st.text_area("Describe your maintenance complaint", key="complaint_text", height=120,
                 placeholder="Example: My bathroom ceiling is leaking continuously.")

    if st.button("Predict Complaint", type="primary"):
        text = st.session_state["complaint_text"]
        if not text.strip():
            st.warning("Please enter a complaint first.")
        else:
            result = analyze_complaint(bundle, text)
            add_to_history(result)
            st.session_state.last_result = result

    if st.session_state.get("last_result"):
        show_result(st.session_state.last_result)


def page_history():
    df = st.session_state.history
    st.markdown("## Complaint History")
    if df.empty:
        st.info("No complaints yet. Predict a complaint to see it here.")
        return
    view = df[["Complaint ID", "Complaint", "Category", "Priority", "Department"]]
    st.dataframe(view, hide_index=True)
    st.download_button("Download history as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="fixmate_complaint_history.csv", mime="text/csv")
    st.button("Clear history", on_click=clear_history)


def page_analytics():
    df = st.session_state.history
    st.markdown("## Analytics")
    if df.empty:
        st.info("No data yet. Predict some complaints (or load demo complaints on the Dashboard).")
        return
    c1, c2 = st.columns(2)
    with c1:
        bar_chart(category_counts(df), BLUE, "Complaints by category")
    with c2:
        bar_chart(urgency_counts(df), [URGENCY_COLORS[u] for u in URGENCY_ORDER], "Complaints by urgency")
    bar_chart(department_counts(df), "#0B2A6F", "Department-wise complaint distribution")


def page_model():
    b = get_model()
    st.markdown("## Model Performance")
    st.caption("Logistic Regression + TF-IDF. All numbers are calculated live on the held-out test set.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Test accuracy", f"{b['accuracy'] * 100:.1f}%", BLUE, "on unseen complaints")
    with c2:
        metric_card("Dataset size", b["n_total"], BLUE, "labelled complaints")
    with c3:
        metric_card("Training set", b["n_train"], BLUE, "80% of data")
    with c4:
        metric_card("Test set", b["n_test"], BLUE, "20% of data")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        cm = b["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CATEGORIES)))
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_xticklabels(CATEGORIES, rotation=40, ha="right")
        ax.set_yticklabels(CATEGORIES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "#0F172A")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Precision / recall / F1")
        st.dataframe(b["report_df"])
        st.markdown("#### Training data per category")
        st.bar_chart(b["class_counts"])
    st.info("The dataset is small (made for learning), so accuracy can change if you edit dataset.csv. "
            "With more real complaints the model would become more reliable.")


def page_how():
    st.markdown("## How It Works")
    st.markdown("""
<div class="card"><h4>Pipeline</h4>
<b>Complaint text</b> → Lowercase → Remove punctuation → Tokenize → Remove stop words → Stem →
<b>TF-IDF vectors</b> → <b>Logistic Regression</b> → <b>Category</b><br><br>
In parallel, the original text goes through simple <b>IF-THEN rules</b> to decide <b>Urgency</b>.
Category → <b>Department</b> (lookup table). Category + Urgency → <b>Recommended action</b>.</div>

<div class="card"><h4>Machine learning vs rules</h4>
<b>Category</b> is predicted by a trained <b>machine-learning model</b>.<br>
<b>Urgency</b> is a <b>separate, rule-based component</b>: keyword rules such as fire, gas leakage, flooding,
major water leakage, exposed wires and structural damage → High; AC, tap, drainage, broken furniture → Medium;
bulb replacement, minor cleaning, cosmetic repairs → Low. It is not machine learning, so every decision is
easy to explain.</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("nav", PAGES[0])

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=90)
        st.markdown("### FixMate AI")
        st.caption("Smart Maintenance Ticket Classifier")
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.markdown("---")
        st.caption("NLP + Machine Learning\nRuns fully offline")

    page = st.session_state["nav"]
    {
        "Dashboard": page_dashboard,
        "Predict Complaint": page_predict,
        "Complaint History": page_history,
        "Analytics": page_analytics,
        "Model Performance": page_model,
        "How It Works": page_how,
    }[page]()


main()
