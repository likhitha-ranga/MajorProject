# %% [markdown]
# Importing the Dependencies

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# %% [markdown]
# Data Collection and Processing

# %%
# Loading the diabetes dataset to a Pandas DataFrame
diabetes_dataset = pd.read_csv('Diabetics\diabetics.csv')

# %%
# Checking the first 5 rows of the dataset
diabetes_dataset.head()

# %%
# Checking for missing values
diabetes_dataset.isnull().sum()

# %% [markdown]
# Splitting the Features and Target

# %%
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# %%
# Normalizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# %% [markdown]
# Handling Class Imbalance (Using SMOTE)

# %%
smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

# %% [markdown]
# Splitting the Data into Training Data & Test Data

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %% [markdown]
# Hyperparameter Tuning for Random Forest

# %%
# Define the parameter grid
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Create the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                               n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
rf_random.fit(X_train, Y_train)

# %% [markdown]
# Best Parameters from Tuning

# %%
print("Best parameters found: ", rf_random.best_params_)

# %% [markdown]
# Building the Optimized Random Forest Model

# %%
# Use the best parameters for the model
best_rf = rf_random.best_estimator_

# Train the optimized model
best_rf.fit(X_train, Y_train)

# %% [markdown]
# Model Evaluation

# %%
# Evaluate on training data
train_predictions = best_rf.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)

# Evaluate on test data
test_predictions = best_rf.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)

# Print accuracy and detailed classification report
print(f"Accuracy on Training Data: {train_accuracy:.2f}")
print(f"Accuracy on Test Data: {test_accuracy:.2f}")
print("\nClassification Report (Test Data):")
print(classification_report(Y_test, test_predictions))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_test, test_predictions))

# %% [markdown]
# Building a Predictive System

# %%
input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)  # Example input

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Normalize the input data
input_data_normalized = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = best_rf.predict(input_data_normalized)
if prediction[0] == 0:
    print('The Person is not Diabetic')
else:
    print('The Person is Diabetic')

