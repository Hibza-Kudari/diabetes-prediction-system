
import streamlit as st
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("models/diabetes_model.pkl", "rb"))

# ---------------- SIDEBAR ----------------
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

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='font-size:50px;'>
🩺 Diabetes Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
Predict the likelihood of diabetes using patient health parameters.
""")

# ---------------- DASHBOARD METRICS ----------------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("🤖 Model", "Random Forest")

with metric2:
    st.metric("🎯 Accuracy", "81%")

with metric3:
    st.metric("📊 Features", "8")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    preg = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0,
        max_value=300,
        value=120
    )

    bp = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70
    )

    skin = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.500,
        format="%.3f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.markdown("")

# ---------------- PREDICT BUTTON ----------------
predict_btn = st.button(
    "🔍 Predict Diabetes",
    use_container_width=True
)

# ---------------- PREDICTION ----------------
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

    st.subheader("🩺 Prediction Result")

    if prediction[0] == 1:

        st.error(
            "⚠️ High Risk: Patient is likely Diabetic"
        )

        st.warning(
            "Please consult a healthcare professional for medical advice."
        )

    else:

        st.success(
            "✅ Low Risk: Patient is likely Non-Diabetic"
        )

        st.info(
            "Maintain a healthy lifestyle and regular health checkups."
        )

# ---------------- MODEL INFO ----------------
st.markdown("---")

st.subheader("📊 Model Information")

st.write("""
**Algorithm:** Random Forest Classifier

**Dataset:** Pima Indians Diabetes Dataset

**Features Used:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Prediction Type:** Binary Classification
""")

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "Built with ❤️ using Python, Scikit-Learn and Streamlit"
)

