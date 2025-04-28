import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier, Pool
import xgboost as xgb

# ----------------- Load Data -------------------
data = pd.read_csv('Parkinson/parkinsons.csv')
X = data.drop(columns=['status', 'name'], axis=1)
Y = data['status']

# ----------------- Train-Test Split -------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# ----------------- Feature Selection for RF & XGB -------------------
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()]

# Convert selected arrays to DataFrames for SHAP compatibility
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

# ----------------- SMOTE -------------------
smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_selected_df, Y_train)

# ----------------- CatBoost (Full Features) -------------------
cb_pool = Pool(X_train, label=Y_train)
catboost_model = CatBoostClassifier(
    verbose=0,
    random_seed=42,
    loss_function='Logloss',
    eval_metric='Accuracy'
)
catboost_model.fit(cb_pool)
cat_preds = catboost_model.predict(X_test)

# ----------------- Random Forest -------------------
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True
)
rf_model.fit(X_train_resampled, Y_train_resampled)
rf_preds = rf_model.predict(X_test_selected_df)

# ----------------- XGBoost -------------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6
)
xgb_model.fit(X_train_resampled, Y_train_resampled)
xgb_preds = xgb_model.predict(X_test_selected_df)

# ----------------- Evaluation Utilities -------------------
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0][0], cm[0][1]
    return tn / (tn + fp)

def evaluate_model(name, y_true, y_pred):
    print(f"\n\033[1mModel: {name}\033[0m")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ----------------- Model Evaluation -------------------
evaluate_model("CatBoost", Y_test, cat_preds)
evaluate_model("Random Forest", Y_test, rf_preds)
evaluate_model("XGBoost", Y_test, xgb_preds)

# ----------------- Evaluation Metrics Table -------------------
results = {
    'Model': ['CatBoost', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        round(accuracy_score(Y_test, cat_preds), 4),
        round(accuracy_score(Y_test, rf_preds), 4),
        round(accuracy_score(Y_test, xgb_preds), 4)
    ],
    'Precision': [
        round(precision_score(Y_test, cat_preds), 4),
        round(precision_score(Y_test, rf_preds), 4),
        round(precision_score(Y_test, xgb_preds), 4)
    ],
    'Sensitivity (Recall)': [
        round(recall_score(Y_test, cat_preds), 4),
        round(recall_score(Y_test, rf_preds), 4),
        round(recall_score(Y_test, xgb_preds), 4)
    ],
    'Specificity': [
        round(specificity_score(Y_test, cat_preds), 4),
        round(specificity_score(Y_test, rf_preds), 4),
        round(specificity_score(Y_test, xgb_preds), 4)
    ],
    'F1 Score': [
        round(f1_score(Y_test, cat_preds), 4),
        round(f1_score(Y_test, rf_preds), 4),
        round(f1_score(Y_test, xgb_preds), 4)
    ]
}
results_df = pd.DataFrame(results)
print("\n========== Evaluation Metrics Table ==========")
print(results_df.to_string(index=False))

# ----------------- Bar Chart -------------------
results_df.set_index('Model')[['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score']].plot(
    kind='bar', figsize=(10, 6), colormap='Accent', edgecolor='black'
)
plt.title('Model Comparison on Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ----------------- Save Models -------------------
joblib.dump(catboost_model, 'catboost_parkinson_pipeline.pkl')

# ----------------- Sample Prediction -------------------
sample_input = (
    123.0, 132.0, 123.0, 0.0025, 0.0012, 0.0013, 0.0018,
    0.005, 0.012, 0.25, 0.01, 0.015, 0.013, 0.03,
    0.011, 0.001, 18.0, 0.3, 0.45, -3.0,
    0.05, 0.002
)
input_array = np.asarray(sample_input).reshape(1, -1)
input_df = pd.DataFrame(input_array, columns=X.columns)

# Predict using CatBoost
prediction = catboost_model.predict(input_df)

print("\nPrediction on Sample Input:")
print("Predicted Class:", prediction[0])
if prediction[0] == 0:
    print("\033[1mThe Person does NOT have Parkinson’s\033[0m")
else:
    print("\033[1mThe Person HAS Parkinson’s\033[0m")

# ----------------- Correlation Heatmap -------------------
corr_matrix = data.drop(columns=['name']).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ----------------- SHAP Explainability for CatBoost -------------------
explainer_cb = shap.TreeExplainer(catboost_model)
shap_values_cb = explainer_cb.shap_values(X_test)
shap.summary_plot(shap_values_cb, X_test, feature_names=X.columns)

# Force plot for input sample
shap_values_input_cb = explainer_cb.shap_values(input_df)
shap.force_plot(
    explainer_cb.expected_value,
    shap_values_input_cb[0],
    input_df.iloc[0],
    matplotlib=True,
    feature_names=X.columns
)
