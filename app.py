# 📁 Project: FairSight - Bias & Fairness Auditor

# ─────────────────────────────────────────────────────────
# STEP 1: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────
# STEP 2: Load Dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/OmBhonde/fairsight-bias-auditor/main/adult.csv'
    data = pd.read_csv(url)
    data = data.dropna()
    return data

# ─────────────────────────────────────────────────────────
# STEP 3: Preprocess
@st.cache_data
def preprocess(df):
    df = df.copy()
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# ─────────────────────────────────────────────────────────
# STEP 4: Train Model
@st.cache_data
def train_model(df, target='income', sensitive='sex'):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test, X_test[sensitive]

# ─────────────────────────────────────────────────────────
# STEP 5: Fairness Evaluation
@st.cache_data
def evaluate_fairness(model, X_test, y_test, sensitive_feature):
    y_pred = model.predict(X_test)
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    disparity = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_feature)
    return mf.by_group, disparity

# ─────────────────────────────────────────────────────────
# STEP 6: Streamlit UI
st.title("🔍 FairSight - Bias & Fairness Auditor")
data = load_data()
df, encoders = preprocess(data)
model, X_test, y_test, sensitive = train_model(df)

st.write("### Dataset Sample")
st.dataframe(data.head())

st.write("### Fairness Evaluation")
group_metrics, disparity = evaluate_fairness(model, X_test, y_test, sensitive)
st.write("Group-wise Metrics")
st.dataframe(group_metrics)
st.write(f"Demographic Parity Difference: `{disparity:.4f}`")

st.write("### SHAP Explanation (Disabled)")
st.warning("SHAP visualization is disabled due to environment limitations (micropip issue). Run locally to enable this feature.")

# st.write("### SHAP Explanation")
# try:
#     import shap
#     @st.cache_data
#     def shap_explanation(model, X_test):
#         explainer = shap.Explainer(model, X_test)
#         shap_values = explainer(X_test)
#         return shap_values
#     shap_values = shap_explanation(model, X_test)
#     fig, ax = plt.subplots()
#     shap.plots.beeswarm(shap_values, max_display=10)
#     st.pyplot(fig)
# except ImportError as e:
#     st.error("SHAP module could not be loaded. Please ensure it is installed in your environment.")

st.success("Fairness audit complete ✅")
