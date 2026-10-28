
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# Get the folder containing this Python script
APP_DIR = Path(__file__).resolve().parent

st.set_page_config(layout="wide")

# Load the trained model
with open(APP_DIR / "best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler
with open(APP_DIR / "scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Define the input features
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "Geography_France", "Geography_Germany",
    "Geography_Spain", "Gender_Female", "Gender_Male",
    "HasCrCard_0", "HasCrCard_1", "IsActiveMember_0",
    "IsActiveMember_1"
]

# Columns requiring scaling
scale_vars = [
    "CreditScore", "EstimatedSalary", "Tenure",
    "Balance", "Age", "NumOfProducts"
]

# Default input values
default_values = [
    600, 30, 2, 8000, 2, 60000,
    True, False, False, True, False,
    False, True, False, True
]

# Sidebar
# Use the exact capitalization of the filenames in GitHub.
st.sidebar.image(
    str(APP_DIR / "Pic 1.PNG"),
    use_container_width=True
)
st.sidebar.header("User Inputs")

# Collect user inputs
user_inputs = {}

for i, feature in enumerate(feature_names):
    if feature in scale_vars:
        user_inputs[feature] = st.sidebar.number_input(
            feature,
            value=float(default_values[i]),
            step=1.0
        )
    else:
        user_inputs[feature] = st.sidebar.checkbox(
            feature,
            value=default_values[i]
        )

# Convert inputs to a DataFrame
input_data = pd.DataFrame([user_inputs])

# Apply the saved scaler
input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(
    input_data[scale_vars]
)

# App header
st.image(
    str(APP_DIR / "Pic 2.PNG"),
    use_container_width=True
)
st.title("Customer Churn Prediction")

# Page layout
left_col, right_col = st.columns(2)

# Feature importance
with left_col:
    st.header("Feature Importance")

    feature_importance_df = pd.read_excel(
        APP_DIR / "feature_importance.xlsx",
        usecols=["Feature", "Feature Importance Score"]
    )

    fig = px.bar(
        feature_importance_df.sort_values(
            by="Feature Importance Score",
            ascending=True
        ),
        x="Feature Importance Score",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        labels={
            "Feature Importance Score": "Importance",
            "Feature": "Features"
        },
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# Churn prediction
with right_col:
    st.header("Prediction")

    if st.button("Predict"):
        probabilities = model.predict_proba(input_data_scaled)[0]
        prediction = model.predict(input_data_scaled)[0]

        prediction_label = (
            "Churned" if prediction == 1 else "Retain"
        )

        st.subheader(f"Predicted Value: {prediction_label}")
        st.write(f"Churn probability: {probabilities[1]:.2%}")
        st.write(f"Retention probability: {probabilities[0]:.2%}")
        st.markdown(f"### Output: **{prediction_label}**")
