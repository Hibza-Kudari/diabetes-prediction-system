import streamlit as st
import pandas as pd
import pickle

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Load Model
model = pickle.load(open("models/diabetes_model.pkl", "rb"))

# Sidebar
st.sidebar.title("🩺 Diabetes Prediction")

st.sidebar.markdown("---")

st.sidebar.info("""
### About Project

This project uses Machine Learning
to predict whether a patient is likely
to have diabetes.

**Model Used:** Random Forest

**Dataset:** Pima Indians Diabetes Dataset

**Accuracy:** 81%
""")

# Main Title
st.title("🩺 Diabetes Prediction System")

st.markdown(
"""
Predict the likelihood of diabetes using
patient health parameters.
"""
)

st.markdown("---")

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, value=0)
    bp = st.number_input("Blood Pressure", min_value=0, value=0)
    skin = st.number_input("Skin Thickness", min_value=0, value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0, value=0)
    bmi = st.number_input("BMI", min_value=0.0, value=0.0)
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.0,
        format="%.3f"
    )
    age = st.number_input("Age", min_value=0, value=0)

st.markdown("")

# Centered button
predict_btn = st.button("🔍 Predict Diabetes")

if predict_btn:

    data = pd.DataFrame([{
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    prediction = model.predict(data)

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Patient is likely Diabetic")
    else:
        st.success("✅ Low Risk: Patient is likely Non-Diabetic")

# Footer
st.markdown("---")

st.subheader("📊 Model Information")

st.write("""
- Algorithm: Random Forest Classifier
- Dataset: Pima Indians Diabetes Dataset
- Features Used: 8
- Prediction Type: Binary Classification
""")