import streamlit as st
import pandas as pd
import joblib

# Load scaler and models
scaler = joblib.load("scaler.pkl")
expected_features = scaler.feature_names_in_

models = {
    "Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "Best Random Forest": joblib.load("best_random_forest_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "Best SVM": joblib.load("best_svm_model.pkl"),
}

# Streamlit UI
st.title("Credit Risk Prediction System")
st.write("Enter customer details to assess credit risk.")

# Model selector
model_choice = st.sidebar.selectbox("Choose a model", list(models.keys()))

# Input fields
age = st.slider("Age", 18, 100, 35)
job = st.selectbox("Job (0 = unemployed, 3 = highly skilled)", [0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=100, max_value=20000, value=1000)
duration = st.slider("Loan Duration (months)", 4, 72, 24)

sex = st.selectbox("Sex", ["male", "female"])
housing = st.selectbox("Housing", ["own", "rent", "free", "no_info"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich", "no_info"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "no_info"])
purpose = st.selectbox("Purpose", [
    "radio/TV", "education", "furniture/equipment", "car", "business",
    "domestic appliances", "repairs", "vacation/others"
])

# Start with basic features
user_input = {
    'Age': age,
    'Job': job,
    'Credit amount': credit_amount,
    'Duration': duration,
    f'Sex_{sex}': 1,
    f'Housing_{housing}': 1,
    f'Saving accounts_{saving_accounts}': 1,
    f'Checking account_{checking_account}': 1,
    f'Purpose_{purpose}': 1
}

# Initialize full input vector with 0s
input_dict = {feat: 0 for feat in expected_features}
for key, value in user_input.items():
    if key in input_dict:
        input_dict[key] = value

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Scale and Predict
input_scaled = scaler.transform(input_df)

if st.button("Predict Credit Risk"):
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    result = "Bad Credit Risk ❌" if prediction == 1 else "Good Credit Risk ✅"
    st.subheader(f"Prediction: {result}")
