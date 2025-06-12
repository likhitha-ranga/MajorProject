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
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ----------------- Load Data -------------------
data = pd.read_csv('Diabetics/diabetes_prediction_dataset.csv')

# Define features and target (same as your second snippet)
features = ['age', 'gender', 'bmi', 'smoking_history', 'hypertension',
            'heart_disease', 'HbA1c_level', 'blood_glucose_level']
target = 'diabetes'

data.dropna(inplace=True)

X = data[features]
Y = data[target]

# ----------------- Train-Test Split -------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42)

# Categorical features for CatBoost
cat_features = ['gender', 'smoking_history']

# ----------------- Handle class imbalance using SMOTE -------------------
# SMOTE expects numeric data only, so apply it after encoding categoricals or skip SMOTE here for CatBoost
# Let's encode categoricals temporarily for SMOTE, then revert for CatBoost Pools

X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

# Simple encoding for SMOTE (label encoding)
for col in cat_features:
    X_train_enc[col] = X_train_enc[col].astype('category').cat.codes
    X_test_enc[col] = X_test_enc[col].astype('category').cat.codes

smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train_enc, Y_train)

# Now recover original categorical values after SMOTE for CatBoost Pool
# We can map encoded back to original categories or just recreate a DataFrame with categorical dtype

X_train_res = pd.DataFrame(X_train_res, columns=X_train_enc.columns)
for col in cat_features:
    # Map codes back to original categories by inverse transform
    # Using categories from training data
    cat_map = dict(enumerate(X_train[col].astype('category').cat.categories))
    X_train_res[col] = X_train_res[col].map(cat_map).astype('category')

# ----------------- Prepare CatBoost Pool -------------------
train_pool = Pool(data=X_train_res, label=Y_train_res, cat_features=cat_features)
test_pool = Pool(data=X_test, label=Y_test, cat_features=cat_features)

# ----------------- Train CatBoost Model -------------------
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42,
    early_stopping_rounds=50
)
model.fit(train_pool, eval_set=test_pool)

cat_preds = model.predict(test_pool)

# ----------------- Feature Selection for RF & XGB -------------------
# For RF & XGB, encode categorical to numeric since they don't handle categoricals natively
X_train_res_rf = X_train_res.copy()
X_test_rf = X_test.copy()

for col in cat_features:
    X_train_res_rf[col] = X_train_res_rf[col].astype('category').cat.codes
    X_test_rf[col] = X_test_rf[col].astype('category').cat.codes

selector = SelectKBest(score_func=f_classif, k=5)
X_train_selected = selector.fit_transform(X_train_res_rf, Y_train_res)
X_test_selected = selector.transform(X_test_rf)
selected_features = X_train_res_rf.columns[selector.get_support()]

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
rf_model.fit(X_train_scaled, Y_train_res)
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
xgb_model.fit(X_train_scaled, Y_train_res)
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

# ----------------- Feature Importance for CatBoost -------------------
feature_importances = model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in CatBoost Model")
plt.tight_layout()
plt.show()

# ----------------- Save CatBoost Model -------------------
joblib.dump(model, 'catboost_diabetes_pipeline.pkl')

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
input_df = pd.DataFrame([sample_input])

cat_prediction = model.predict(input_df)[0]
print("\nPrediction on Sample Input:")
print("Predicted Class:", cat_prediction)
if cat_prediction == 0:
    print("\033[1mThe Person does NOT have Diabetes\033[0m")
else:
    print("\033[1mThe Person HAS Diabetes\033[0m")

# ----------------- Correlation Heatmap -------------------
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# For barplot (remove palette or add hue)
sns.barplot(x=feature_importances, y=feature_names)  # simplest fix

# For correlation heatmap
corr_matrix = data.select_dtypes(include=[np.number]).corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# ----------------- SHAP Explainability -------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

shap_values_input = explainer.shap_values(input_df)
shap.force_plot(
    explainer.expected_value,
    shap_values_input[0],
    input_df.iloc[0],
    feature_names=X.columns,
    matplotlib=True
)
