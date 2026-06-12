import pickle
import pandas as pd

model = pickle.load(open("models/diabetes_model.pkl", "rb"))

sample = pd.DataFrame([{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 79,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}])

prediction = model.predict(sample)

print(prediction)

if prediction[0] == 1:
    print("Diabetic")
else:
    print("Not Diabetic")