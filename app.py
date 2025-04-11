import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="wide")
st.title("📊 Bank Term Deposit Predictor")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/OmBhonde/fairsight-bias-auditor/main/bank.csv"
    data = pd.read_csv(url, sep=';')

    # Clean object columns
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype(str).str.strip()

    # DEBUG: Show target value distribution before filtering
    st.write("🧪 Unique values in 'y' BEFORE processing:", data['y'].unique())

    # Filter out invalid values if any
    valid_targets = ['yes', 'no']
    data = data[data['y'].isin(valid_targets)]

    # DEBUG: Show class distribution
    st.write("✅ Class counts AFTER filtering:", data['y'].value_counts())

    return data

def encode_features(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        if col != 'y':
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    df_encoded['y'] = df_encoded['y'].apply(lambda x: 1 if x == 'yes' else 0)
    return df_encoded

def train_model(df):
    X = df.drop('y', axis=1)
    y = df['y']

    st.write("📊 Target class distribution (y):", y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Target variable 'y' has fewer than 2 unique classes after cleaning. Please check your dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.write("### Model Evaluation")
    st.text("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    st.text("\nClassification Report:\n" + classification_report(y_test, y_pred))

# Main logic
df = load_data()
df_encoded = encode_features(df)
model, X_test, y_test = train_model(df_encoded)
evaluate_model(model, X_test, y_test)
