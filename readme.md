# 🩺 Diabetes Prediction System

## 📌 Overview

The Diabetes Prediction System is a Machine Learning project that predicts whether a patient is likely to have diabetes based on health-related parameters.

The project uses the Pima Indians Diabetes Dataset and implements multiple supervised learning algorithms including Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM). A Streamlit web application is also developed to provide real-time predictions through a user-friendly interface.

---

## 🚀 Features

* Data preprocessing and cleaning
* Handling missing values
* Exploratory Data Analysis (EDA)
* Multiple Machine Learning models
* Model performance comparison
* Real-time diabetes prediction
* Interactive Streamlit web application
* Model serialization using Pickle

---

## 📊 Dataset

Dataset: Pima Indians Diabetes Dataset

### Features

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| Pregnancies              | Number of pregnancies      |
| Glucose                  | Blood glucose level        |
| BloodPressure            | Blood pressure measurement |
| SkinThickness            | Skin fold thickness        |
| Insulin                  | Insulin level              |
| BMI                      | Body Mass Index            |
| DiabetesPedigreeFunction | Diabetes hereditary score  |
| Age                      | Age of the patient         |

### Target Variable

* 0 → Non-Diabetic
* 1 → Diabetic

---

## 🧠 Machine Learning Models Used

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)

---

## 📈 Results

| Model               | Accuracy                  |
| ------------------- | ------------------------- |
| Logistic Regression | 75%                       |
| Decision Tree       | (Update with your result) |
| Random Forest       | (Update with your result) |
| SVM                 | (Update with your result) |

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Pickle

---

## 📂 Project Structure

Diabetes_Prediction/

├── dataset/

│   └── diabetes.csv

├── models/

│   └── diabetes_model.pkl

├── notebook/

├── main.py

├── predict.py

├── app.py

├── requirements.txt

└── README.md

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 🖥️ Streamlit Application

The application allows users to enter health parameters and receive instant diabetes predictions using the trained Random Forest model.

---

## 📸 Screenshots

### Home Page

(Add Streamlit homepage screenshot here)

### Prediction Result

(Add prediction output screenshot here)

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Advanced feature engineering
* Deep Learning implementation
* Cloud deployment
* User authentication
* Medical report generation

---

## 👨‍💻 Author

Developed as an AI/ML project using Machine Learning and Streamlit.
