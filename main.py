import pandas as pd

df = pd.read_csv("dataset/diabetes.csv")

columns = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]

# Replace 0 with median
for col in columns:
    df[col] = df[col].replace(0, df[col].median())

print(df.describe())

# Features and Target

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nLogistic Regression Accuracy:")
print(accuracy)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_predictions = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nDecision Tree Accuracy:")
print(dt_accuracy)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\nRandom Forest Accuracy:")
print(rf_accuracy)

from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nSVM Accuracy:")
print(svm_accuracy)

print("\n----- Model Comparison -----")

print("Logistic Regression :", accuracy)
print("Decision Tree       :", dt_accuracy)
print("Random Forest       :", rf_accuracy)
print("SVM                 :", svm_accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, rf_predictions)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))

import pickle

pickle.dump(rf, open("models/diabetes_model.pkl", "wb"))

print("Model Saved Successfully!")

