import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ----------------- Load Data -------------------
data = pd.read_csv('Diabetics/diabetics.csv')
X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

scaler = StandardScaler()
X1 = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X1, Y1 = smote.fit_resample(X1, Y)

# ----------------- Train-Test Split -------------------
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, stratify=Y1, random_state=42)

# ----------------- SMOTE -------------------
smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# ----------------- CatBoost (All Features) -------------------
catboost_pipeline = Pipeline([
    ('classifier', CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    ))
])

catboost_param_grid = {
    'classifier__iterations': [500, 1000, 1500],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__depth': [6, 8, 10],
    'classifier__l2_leaf_reg': [1, 3, 5],
    'classifier__border_count': [32, 50],
}

catboost_search = RandomizedSearchCV(
    estimator=catboost_pipeline,
    param_distributions=catboost_param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=100,
    n_jobs=-1
)

catboost_search.fit(X_train, Y_train)
best_catboost = catboost_search.best_estimator_
cat_preds = best_catboost.predict(X_test)

# ----------------- Feature Selection for RF & XGB -------------------
selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train_resampled, Y_train_resampled)
X_test_selected = selector.transform(X_test)
selected_features = X.columns[selector.get_support()]

# ----------------- Scaling -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# ----------------- Random Forest -------------------
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True
)
rf_model.fit(X_train_scaled, Y_train_resampled)
rf_preds = rf_model.predict(X_test_scaled)

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
xgb_model.fit(X_train_scaled, Y_train_resampled)
xgb_preds = xgb_model.predict(X_test_scaled)

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
    kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black'
)
plt.title('Model Comparison on Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


best_catboost=joblib.load('catboost_diabetes_pipeline.pkl')

# ----------------- Sample Prediction -------------------
input_data = (1, 85, 66, 29, 0, 26.6, 0.351, 31)  # Expected to be class 0
input_array = np.asarray(input_data).reshape(1, -1)
cat_prediction = best_catboost.predict(input_array)

print("\nPrediction on Sample Input:")
print("Predicted Class:", cat_prediction[0])
if cat_prediction[0] == 0:
    print("\033[1mThe Person does NOT have Diabetes\033[0m")
else:
    print("\033[1mThe Person HAS Diabetes\033[0m")

# ----------------- Correlation Heatmap -------------------
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ----------------- SHAP Explainability -------------------
explainer = shap.TreeExplainer(best_catboost.named_steps['classifier'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

shap_values_input = explainer.shap_values(input_array)
shap.force_plot(
    explainer.expected_value,
    shap_values_input[0],
    input_array[0],
    feature_names=X.columns,
    matplotlib=True
)
