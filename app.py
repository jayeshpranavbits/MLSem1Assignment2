import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef
)

# -----------------------------
# Page config + header
# -----------------------------
st.set_page_config(page_title="Dry Bean ML Classifier", layout="wide")
st.title("Dry Bean Classification Dashboard")
st.caption("Multi-class classification (7 bean types) using 6 ML models with required evaluation metrics.")
st.caption("Done by: Student: Jayesh Pranav - 2025AA05097 | Course: Machine Learning Semester 1 Assignment 2")
st.markdown(
    """
**Models:** Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, Random Forest, XGBoost  
**Metrics:** Accuracy, Precision (Macro), Recall (Macro), F1 (Macro), MCC, ROC-AUC (OvR Macro)
"""
)

st.divider()

# -----------------------------
# Paths + model mapping
# -----------------------------
MODEL_DIR = "model"
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes_gaussian.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib"
}

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

selected = st.sidebar.selectbox("Select Model", list(model_files.keys()))
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload rules**")
st.sidebar.markdown("- CSV should contain the **same 16 feature columns**")
st.sidebar.markdown("- Target column preferred name: **Class**")
st.sidebar.markdown("- If labels missing, app runs in **Prediction Mode**")

# -----------------------------
# Utilities
# -----------------------------
def compute_metrics_multiclass(y_true, y_pred, y_proba, n_classes):
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (Macro)": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall (Macro)": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 (Macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
        out["ROC-AUC (OvR Macro)"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    else:
        out["ROC-AUC (OvR Macro)"] = None
    return out

def kpi_row(metrics: dict):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cols = [c1, c2, c3, c4, c5, c6]
    for col, (k, v) in zip(cols, metrics.items()):
        if v is None:
            col.metric(k, "—")
        else:
            col.metric(k, f"{v:.4f}")

def safe_stop(msg: str):
    st.info(msg)
    st.stop()

# -----------------------------
# Pre-check required files
# -----------------------------
if not os.path.exists(ENCODER_PATH):
    st.error("Missing `model/label_encoder.joblib`. Please commit the `model/` folder to GitHub.")
    st.stop()

model_path = os.path.join(MODEL_DIR, model_files[selected])
if not os.path.exists(model_path):
    st.error(f"Missing model file: `{model_path}`. Commit it to GitHub (or shrink/compress if too large).")
    st.stop()

le = joblib.load(ENCODER_PATH)
class_names = list(le.classes_)
n_classes = len(class_names)
model = joblib.load(model_path)

# -----------------------------
# Main: Upload handling
# -----------------------------
if uploaded is None:
    safe_stop("Upload a CSV from the sidebar to begin.")

df = pd.read_csv(uploaded)

# Determine label column
target_col = "Class" if "Class" in df.columns else df.columns[-1]

# Basic preview
st.subheader("Uploaded Data Preview")
st.write(f"Detected target column: **{target_col}**")
st.dataframe(df.head(10), use_container_width=True)

# Separate X/y (y optional)
if target_col in df.columns:
    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()
else:
    X = df.copy()
    y_raw = pd.Series([np.nan] * len(df))

# Ensure numeric features
X = X.apply(pd.to_numeric, errors="coerce")
if X.isna().any().any():
    st.warning("Some feature values are not numeric. Dropping rows with NaNs.")
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y_raw = y_raw.loc[mask].reset_index(drop=True)

# Clean label column
y_raw = y_raw.astype(str).str.strip()
y_raw = y_raw.replace({"": np.nan, "nan": np.nan, "None": np.nan})

st.divider()
st.subheader("Model Output")

# -----------------------------
# Prediction-only mode
# -----------------------------
if y_raw.isna().any():
    st.warning("Labels are missing/blank → running **Prediction Mode** (no metrics shown).")

    preds = model.predict(X)
    pred_labels = le.inverse_transform(preds)

    out = X.copy()
    out["Predicted_Class"] = pred_labels

    st.dataframe(out.head(50), use_container_width=True)

    st.download_button(
        "Download predictions CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="drybean_predictions.csv",
        mime="text/csv"
    )

    with st.expander("Notes"):
        st.markdown(
            """
- To view metrics, include the **true label column** (`Class`) in your CSV.
- Class labels must match training classes exactly:
  `BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA`.
"""
        )
    st.stop()

# -----------------------------
# Evaluation mode
# -----------------------------
try:
    y = le.transform(y_raw)
except Exception:
    st.error("Your uploaded label values do not match the training classes.")
    st.write("Expected classes:", class_names)
    st.stop()

y_pred = model.predict(X)
y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

metrics = compute_metrics_multiclass(y, y_pred, y_proba, n_classes)

st.success(f"Evaluation complete using **{selected}**")

# KPI metrics row
kpi_row(metrics)

# Expanders for deeper outputs
with st.expander("Confusion Matrix"):
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

with st.expander("Classification Report"):
    st.text(classification_report(y, y_pred, target_names=class_names, digits=4, zero_division=0))

st.divider()
st.caption("Made for academic submission for Machine Learning - Assignment 2: model comparison, specified input handling, and required model metrics reporting for test data as per chosen model.")