import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
diabetes_data = pd.read_csv('Diabetics/diabetics.csv')

# Splitting features and target
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# ----------------- Train-Test Split -------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# ----------------- SMOTE -------------------
smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

# ----------------- CatBoost (All Features) -------------------
pipeline = Pipeline([
    ('classifier', CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    ))
])

# Define hyperparameter grid for CatBoostClassifier (inside pipeline)
param_grid = {
    'classifier__iterations': [500, 1000, 1500],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__depth': [6, 8, 10],
    'classifier__l2_leaf_reg': [1, 3, 5],
    'classifier__border_count': [32, 50],
}

# Perform hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    verbose=100,
    n_jobs=-1
)

# Fit the pipeline
random_search.fit(X_train_resampled, Y_train_resampled)

# Best parameters
print(f"Best parameters from RandomizedSearchCV: {random_search.best_params_}")

# Final trained pipeline
best_pipeline = random_search.best_estimator_

# Predict and evaluate
catboost_preds = best_pipeline.predict(X_test)
catboost_accuracy = accuracy_score(Y_test, catboost_preds)

print(f"CatBoost Test Accuracy: {catboost_accuracy:.4f}")
print("\nCatBoost Classification Report on Test Data:")
print(classification_report(Y_test, catboost_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, catboost_preds))

import joblib
joblib.dump(best_pipeline, 'catboost_diabetes_pipeline.pkl')

