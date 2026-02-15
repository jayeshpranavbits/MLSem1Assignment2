
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

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classification (6 Models)")

st.markdown("""
Upload a CSV with the same **feature columns** used for training, plus a target column (bean class).
If your target column is named `Class`, the app will use it automatically; otherwise it will use the **last column**.
""")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

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

def compute_metrics_multiclass(y_true, y_pred, y_proba, n_classes):
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
        out["AUC_ovr_macro"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    else:
        out["AUC_ovr_macro"] = None
    return out

col1, col2 = st.columns([1, 2])

with col1:
    selected = st.selectbox("Select Model", list(model_files.keys()))

if uploaded is not None:
    df = pd.read_csv(uploaded)

    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        st.warning("Some feature values could not be parsed as numeric. Dropping rows with NaNs.")
        mask = ~X.isna().any(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y_raw = y_raw.loc[mask].reset_index(drop=True)

    if not os.path.exists(ENCODER_PATH):
        st.error("label_encoder.joblib not found in model/. Please include it in your repo.")
        st.stop()

    le = joblib.load(ENCODER_PATH)
    class_names = list(le.classes_)
    n_classes = len(class_names)

y_raw = y_raw.astype(str).str.strip()
y_raw = y_raw.replace({"": np.nan, "nan": np.nan, "None": np.nan})

if y_raw.isna().any():
    st.warning("Some label values are missing/blank. Running in prediction-only mode (no metrics).")

    model = joblib.load(model_path)
    preds = model.predict(X)

    pred_labels = le.inverse_transform(preds)

    out = X.copy()
    out["Predicted_Class"] = pred_labels
    st.subheader("Predictions")
    st.dataframe(out.head(50))

    st.download_button(
        "Download predictions as CSV",
        out.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )
    st.stop()

y = le.transform(y_raw)

    model_path = os.path.join(MODEL_DIR, model_files[selected])
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Ensure model/ folder is in your repo.")
        st.stop()

    model = joblib.load(model_path)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics_multiclass(y, y_pred, y_proba, n_classes)
    cm = confusion_matrix(y, y_pred)

    with col2:
        st.subheader("Evaluation Metrics")
        st.json(metrics)

        st.subheader("Confusion Matrix")
        st.write(cm)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred, target_names=class_names, digits=4, zero_division=0))
else:
    st.info("Upload a CSV to evaluate the selected model.")
