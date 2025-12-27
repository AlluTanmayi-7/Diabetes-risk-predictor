import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

import visuals

# Load dataset
df = pd.read_csv("Diabetes Data/diabetes.csv")

df.columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# Visualizations
visuals.plot_feature_distributions(df)
visuals.plot_correlation_heatmap(df)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
visuals.plot_confusion_matrix(cm)

# Save model
joblib.dump(model, "model/diabetes_model.pkl")
print("Model saved as diabetes_model.pkl")
