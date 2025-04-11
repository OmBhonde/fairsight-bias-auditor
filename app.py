import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Fairsight Bias Auditor", layout="wide")
st.title("📊 Fairsight Bias Auditor")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/OmBhonde/fairsight-bias-auditor/main/adult.csv'
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(url, names=column_names, skiprows=1)

    # Clean all object columns
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype(str).str.strip()

    # DEBUG: Print unique income values
    st.write("🧪 Unique income values BEFORE filtering:", data['income'].unique())

    # Filter only valid income values
    valid_incomes = ['<=50K', '>50K']
    data = data[data['income'].isin(valid_incomes)]

    # DEBUG: Show count after filtering
    st.write("✅ Class counts AFTER filtering:", data['income'].value_counts())

    return data

def encode_features(df):
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        if col != 'income':
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
    df_encoded['income'] = df_encoded['income'].apply(lambda x: 1 if x == '>50K' else 0)
    return df_encoded

def train_model(df):
    X = df.drop('income', axis=1)
    y = df['income']

    st.write("📊 Target class distribution (y):", y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Target variable 'income' has fewer than 2 unique classes after cleaning. Please check your dataset.")

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
