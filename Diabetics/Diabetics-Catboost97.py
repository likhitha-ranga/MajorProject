import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load dataset
df = pd.read_csv('Diabetics/diabetes_prediction_dataset.csv')

# 2. Define all features
features = ['age', 'gender', 'bmi', 'smoking_history', 'hypertension',
            'heart_disease', 'HbA1c_level', 'blood_glucose_level']
target = 'diabetes'

# 3. Drop rows with missing values
df.dropna(inplace=True)

# 4. Categorical feature names
cat_features = ['gender', 'smoking_history']

# 5. Train-test split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Prepare CatBoost Pool
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# 7. Train CatBoost model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42
)
model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# 8. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. Save model
with open('diabetes_model_full.pkl', 'wb') as f:
    pickle.dump(model, f)
import pandas as pd
import pickle

# Sample input data
sample_input= {
    'age': 28,
    'gender': 'female',
    'bmi': 22.4,
    'smoking_history': 'never',
    'hypertension': 0,
    'heart_disease': 0,
    'HbA1c_level': 5.1,
    'blood_glucose_level': 90
}


# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

# Load the trained model
with open('diabetes_model_full.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
prediction = model.predict(input_df)[0]
output = "Yes" if prediction == 1 else "No"

print("Prediction on Sample Input:")
print("Diabetes:", output)
