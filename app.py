import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.title("🔍 FairSight: Bias Detection in ML")

@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/OmBhonde/fairsight-bias-auditor/main/adult.csv'
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(url, names=column_names, skiprows=1)

    # Clean string columns
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype(str).str.strip()

    # Drop rows with missing or invalid income
    data = data[data['income'].isin(['<=50K', '>50K'])]

    return data

df = load_data()

st.subheader("📊 Raw Dataset Preview")
st.write(df.head())

def preprocess_data(df):
    df_encoded = df.copy()
    le_dict = {}
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        le_dict[col] = le
    return df_encoded, le_dict

df_encoded, label_encoders = preprocess_data(df)

def train_model(df):
    X = df.drop('income', axis=1)
    y = df['income']

    # Debug: Show class distribution
    st.write("🎯 Target class distribution:", y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Target variable 'income' has fewer than 2 unique classes after cleaning.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = train_model(df_encoded)

# Evaluation
y_pred = model.predict(X_test)
st.subheader("📈 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Bias Check
st.subheader("🧪 Bias Check: Accuracy by Gender")
X_test_original = df.loc[X_test.index]
X_test_original['prediction'] = y_pred

gender_accuracy = X_test_original.groupby('sex').apply(
    lambda g: accuracy_score(df_encoded.loc[g.index, 'income'], y_pred[g.index])
)

st.write(gender_accuracy)
