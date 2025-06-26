# -Early-Disease-Risk-Prediction-and-Intervention-Tool-EDRIT-
A real-time ML system for predicting chronic disease risk and recommending preventive actions.
Real-World Scenario:
Think about how insurance companies, hospitals, or personal health platforms like MyFitnessPal could use this:

Risk scores = early interventions.

Doctors or users can view interpretable model outputs (e.g., SHAP).

Real use for preventive healthcare or employee wellness programs.

# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# 2. Preprocess
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("Classification Report:\n", classification_report(y_test, preds))

# 5. Explain with SHAP
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], plot_typ
