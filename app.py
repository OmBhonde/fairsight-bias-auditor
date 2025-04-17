import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Fairsight Bias Auditor", layout="wide")
st.title("📊 Fairsight Bias Auditor")
st.sidebar.header("Bias Audit Settings")
protected_attr = st.sidebar.selectbox("Select Protected Attribute", ['job', 'marital', 'education', 'default', 'housing', 'loan'])


@st.cache_data
def load_data():
<<<<<<< HEAD
    data = pd.read_csv("bank.csv", sep=';')  # local file now
=======
    url = "https://raw.githubusercontent.com/OmBhonde/fairsight-bias-auditor/main/bank.csv"
    data = pd.read_csv(url, sep=';')  # bank.csv uses semicolon separator

    # Clean object columns
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype(str).str.strip()

    st.write("📄 Columns in the dataset:", data.columns.tolist())
    st.write("🧪 Unique values in target BEFORE filtering:", data['y'].unique())

    # Filter rows with valid target values
    valid_targets = ['yes', 'no']
    data = data[data['y'].isin(valid_targets)]

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

# Bias Analysis Section
st.header("🧠 Fairness & Bias Check")

protected_attr = st.selectbox("Choose a sensitive attribute to evaluate bias:", 
                              options=['job', 'marital', 'education', 'default', 'housing', 'loan'])

if protected_attr:
    X_test_copy = X_test.copy()
    X_test_copy['y_true'] = y_test
    X_test_copy['y_pred'] = model.predict(X_test)

    if protected_attr in X_test_copy.columns:
        group_perf = X_test_copy.groupby(protected_attr).apply(
            lambda x: pd.Series({
                'Count': len(x),
                'Accuracy': accuracy_score(x['y_true'], x['y_pred']) * 100
            })
        )

        st.subheader(f"📊 Performance by '{protected_attr}'")
        st.dataframe(group_perf.style.format({"Accuracy": "{:.2f}%"}))

        # ✅ Safe plotting inside the same block
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        group_perf['Accuracy'].plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f'Accuracy by {protected_attr}')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        st.pyplot(fig)

        # ✅ Optional Bias Alert
        max_acc = group_perf['Accuracy'].max()
        min_acc = group_perf['Accuracy'].min()
        gap = max_acc - min_acc
        st.write(f"📉 Accuracy gap between best and worst group: `{gap:.2f}%`")

        if gap > 10:
            st.warning("⚠️ Significant bias detected — consider balancing your dataset or using fairer models.")
        else:
            st.success("✅ No major group-level bias found.")
    else:
        st.error(f"'{protected_attr}' not found in the dataset.")

