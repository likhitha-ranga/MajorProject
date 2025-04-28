# Importing Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Data Collection & Analysis
dataset_path = 'Parkinson/parkinsons.csv'
parkinsons_data = pd.read_csv(dataset_path)

# Display basic dataset information
print(parkinsons_data.head())
print("Dataset Shape:", parkinsons_data.shape)
print(parkinsons_data.info())
print(parkinsons_data.isnull().sum())
print(parkinsons_data.describe())
print(parkinsons_data['status'].value_counts())

# Data Preprocessing
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training - Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
rf_model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = rf_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy:', training_data_accuracy)

X_test_prediction = rf_model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Test Accuracy:', test_data_accuracy)

