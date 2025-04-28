import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool

# Load the dataset
heart_data = pd.read_csv('Heart/heart.csv')

# Splitting features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Define the CatBoost Pool (for better handling of categorical features if any)
train_pool = Pool(data=X_train, label=Y_train)
test_pool = Pool(data=X_test, label=Y_test)

# Initialize the CatBoost Classifier
catboost_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# Train the CatBoost model
catboost_model.fit(train_pool, eval_set=test_pool)

# Predictions
catboost_preds = catboost_model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(Y_test, catboost_preds)
precision = precision_score(Y_test, catboost_preds)
recall = recall_score(Y_test, catboost_preds)
f1 = f1_score(Y_test, catboost_preds)

# Print Classification Report
print("CatBoost Classification Report:")
print(classification_report(Y_test, catboost_preds))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(Y_test, catboost_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for CatBoost')
plt.show()

# Feature Importance Plot
feature_importances = catboost_model.get_feature_importance()
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in CatBoost Model")
plt.show()


# SHAP Explainability
explainer = shap.Explainer(catboost_model, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test)

# Save the trained CatBoost model
catboost_model.save_model("catboost_heart_model.cbm")
