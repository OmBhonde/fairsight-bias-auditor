# 🔍 FairSight - Bias & Fairness Auditor

![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)

FairSight is a Streamlit-based web app that allows data scientists, auditors, and researchers to assess and visualize bias and fairness in machine learning models using the `fairlearn` toolkit.

---

## 🚀 Features

- 📊 Load and preprocess the Adult Income dataset
- 🤖 Train a logistic regression model
- 📈 Evaluate group fairness with `fairlearn` metrics
- ⚖️ View demographic parity difference
- 🚧 Optional SHAP visualizations for interpretability (disabled in sandboxed environments)

---

## 📦 Installation

```bash
git clone https://github.com/OmBhonde/fairsight-bias-auditor.git
cd fairsight-bias-auditor
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧠 Built With

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Fairlearn](https://fairlearn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 💡 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 🙌 Acknowledgements

- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Fairlearn contributors
