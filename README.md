
# Loan Default Prediction Model
A machine learning-based project designed to predict whether a loan applicant is likely to default. This helps financial institutions reduce risk and make data-driven lending decisions.

📌 Project Overview
This project utilizes historical loan applicant data to train a predictive model that classifies whether a person is likely to repay a loan. It leverages data preprocessing, feature engineering, and machine learning classification techniques.

🧠 Features
Binary classification: Predict loan approval or default (Loan_Status)

Data cleaning & imputation

Feature engineering (One-Hot Encoding, Label Encoding)

Model training using Logistic Regression, Random Forest, XGBoost

Model evaluation with accuracy, precision, recall, F1-score

Deployment-ready pipeline (can be extended with Streamlit or Flask)

📂 Folder Structure
kotlin
Copy
Edit
Loan-Predictive-Model/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── models/
│   └── loan_model.pkl
│
├── loan_predictor.py
├── requirements.txt
├── README.md
└── .gitignore
🛠️ Tech Stack
Python (Pandas, NumPy, Scikit-learn, XGBoost)

Jupyter Notebook

Matplotlib & Seaborn (for EDA)

Joblib/Pickle for model saving

🚀 How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/yourusername/loan-predictive-model.git
cd loan-predictive-model
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the notebook
Open EDA_and_Modeling.ipynb in Jupyter or VS Code and execute cells step by step.

🔍 Model Evaluation Metrics
Accuracy: 85% (Random Forest)

Precision: 81%

Recall: 87%

F1 Score: 83%

📊 Sample Input Features
Feature	Type	Description
Gender	Categorical	Male/Female
Married	Categorical	Yes/No
ApplicantIncome	Numerical	Monthly income of applicant
LoanAmount	Numerical	Loan amount requested
Credit_History	Binary	1 (has credit history), 0 (none)
Education	Categorical	Graduate / Not Graduate

✅ Future Enhancements
Deploy as a Streamlit or Flask web app

Incorporate more granular financial history features

Add SHAP/LIME interpretability for model decisions
