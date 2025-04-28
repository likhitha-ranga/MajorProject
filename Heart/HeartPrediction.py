import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE

# ----------------- Load the dataset -------------------
data = pd.read_csv('Heart/heart.csv')

# Split features and target
X = data.drop(columns='target', axis=1)
Y = data['target']

# ----------------- Feature Selection (ANOVA) -------------------
anova_scores, _ = f_classif(X, Y)
selected_indices = np.argsort(anova_scores)[-5:]
X_selected = X.iloc[:, selected_indices]

# ----------------- Train-test split -------------------
X_train_full, X_test_full, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

X_train_sel, X_test_sel, _, _ = train_test_split(X_selected, Y, test_size=0.2, stratify=Y, random_state=42)

# ----------------- SMOTE for RF/XGBoost -------------------
smote = SMOTE(random_state=42)
X_train_sel, Y_train_sel = smote.fit_resample(X_train_sel, Y_train)

# ----------------- Scaling -------------------
scaler = StandardScaler()
X_train_sel = scaler.fit_transform(X_train_sel)
X_test_sel = scaler.transform(X_test_sel)

# ----------------- CatBoost (Full Features) -------------------
train_pool = Pool(data=X_train_full, label=Y_train)
test_pool = Pool(data=X_test_full, label=Y_test)

best_catboost = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)
best_catboost.fit(train_pool, eval_set=test_pool)
cat_preds = best_catboost.predict(X_test_full)

# ----------------- Random Forest -------------------
rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True
)
rf_model.fit(X_train_sel, Y_train_sel)
rf_preds = rf_model.predict(X_test_sel)

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
xgb_model.fit(X_train_sel, Y_train_sel)
xgb_preds = xgb_model.predict(X_test_sel)

# ----------------- Evaluation Section -------------------
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Print detailed report per model
evaluate_model("CatBoost", Y_test, cat_preds)
evaluate_model("Random Forest", Y_test, rf_preds)
evaluate_model("XGBoost", Y_test, xgb_preds)

# Create summary table
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
print("\n========== Evaluation Metrics Table ==========\n")
print(results_df.to_string(index=False))

# ----------------- Bar Chart Comparison -------------------
results_df.set_index('Model')[['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score']].plot(
    kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black'
)
plt.title('Model Comparison on Evaluation Metrics')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# ----------------- Highlight Best Performers -------------------
try:
    from IPython.display import display  # works in Jupyter, Streamlit, etc.
    styled_df = results_df.style.highlight_max(axis=0, color='lightgreen', subset=['Accuracy', 'Precision', 'Sensitivity (Recall)', 'Specificity', 'F1 Score'])
    display(styled_df)
except ImportError:
    print("\n(Optional) DataFrame styling only works in notebooks or Streamlit.")

# ----------------- Save Best Model -------------------
joblib.dump(best_catboost, 'catboost_heart_pipeline.pkl')

# ----------------- Sample Prediction -------------------
# Sample input as a dictionary with feature names matching training data
sample_input = {
    'age': 63,
    'sex': 1,          
    'cp': 3,            
    'trestbps': 145,
    'chol': 233,
    'fbs': 1,
    'restecg': 0,
    'thalach': 150,
    'exang': 1,         
    'oldpeak': 2.3,    
    'slope': 0,
    'ca': 1,            
    'thal': 2         
}


# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

# Predict using CatBoost (trained on full features)
prediction = best_catboost.predict(input_df)

print("\nPrediction on Sample Input:")
print("Predicted Class:", prediction[0])
if prediction[0] == 0:
    print("\033[1mThe Person does NOT have Heart Disease\033[0m")
else:
    print("\033[1mThe Person HAS Heart Disease\033[0m")

# ----------------- Feature Correlation Heatmap -------------------
corr_matrix = pd.DataFrame(X, columns=data.columns[:-1]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ----------------- SHAP Explainability -------------------
best_model = xgb_model  # change to best_catboost if needed
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test_sel)
shap.summary_plot(shap_values, X_test_sel, feature_names=X_selected.columns)

