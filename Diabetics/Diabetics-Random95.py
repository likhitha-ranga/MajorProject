import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# ----------------- Load Data -------------------
data = pd.read_csv('Diabetics/diabetes_prediction_dataset.csv')

features = ['age', 'gender', 'bmi', 'smoking_history', 'hypertension',
            'heart_disease', 'HbA1c_level', 'blood_glucose_level']
target = 'diabetes'

data.dropna(inplace=True)
X = data[features]
Y = data[target]

# ----------------- Train-Test Split -------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42)

# ----------------- Encode Categorical Features -------------------
cat_features = ['gender', 'smoking_history']
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

for col in cat_features:
    X_train_enc[col] = X_train_enc[col].astype('category').cat.codes
    X_test_enc[col] = X_test_enc[col].astype('category').cat.codes

# ----------------- SMOTE for Class Balancing -------------------
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train_enc, Y_train)

# ----------------- Feature Selection -------------------
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train_res, Y_train_res)
X_test_selected = selector.transform(X_test_enc)
selected_features = X_train_enc.columns[selector.get_support()]

# ----------------- Scaling -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# ----------------- Train Random Forest Model -------------------
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True
)
rf_model.fit(X_train_scaled, Y_train_res)
rf_preds = rf_model.predict(X_test_scaled)

# ----------------- Evaluation -------------------
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0][0], cm[0][1]
    return tn / (tn + fp)

print("\n\033[1mRandom Forest Model Evaluation\033[0m")
print(f"Accuracy: {accuracy_score(Y_test, rf_preds):.4f}")
print("Classification Report:")
print(classification_report(Y_test, rf_preds))

cm = confusion_matrix(Y_test, rf_preds)
print("Confusion Matrix:")
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------- Metrics Summary -------------------
metrics = {
    'Accuracy': accuracy_score(Y_test, rf_preds),
    'Precision': precision_score(Y_test, rf_preds),
    'Recall (Sensitivity)': recall_score(Y_test, rf_preds),
    'Specificity': specificity_score(Y_test, rf_preds),
    'F1 Score': f1_score(Y_test, rf_preds)
}

results_df = pd.DataFrame([metrics], index=['Random Forest']).round(4)
print("\n========== Evaluation Metrics Table ==========")
print(results_df)

# ----------------- Bar Chart -------------------
results_df.T.plot(kind='bar', legend=False, figsize=(8, 6), color='skyblue', edgecolor='black')
plt.title("Random Forest - Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ----------------- Sample Prediction -------------------
sample_input = {
    'age': 28,
    'gender': 'female',
    'bmi': 22.4,
    'smoking_history': 'never',
    'hypertension': 0,
    'heart_disease': 0,
    'HbA1c_level': 7,
    'blood_glucose_level': 90
}
sample_df = pd.DataFrame([sample_input])

# Encode categorical
for col in cat_features:
    sample_df[col] = sample_df[col].astype('category').cat.codes

# Select and scale
sample_selected = selector.transform(sample_df[selected_features])
sample_scaled = scaler.transform(sample_selected)

# Predict
rf_prediction = rf_model.predict(sample_scaled)[0]
print("\nPrediction on Sample Input:")
print("Predicted Class:", rf_prediction)
if rf_prediction == 0:
    print("\033[1mThe Person does NOT have Diabetes\033[0m")
else:
    print("\033[1mThe Person HAS Diabetes\033[0m")
