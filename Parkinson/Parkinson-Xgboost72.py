# Importing Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
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

# Model Training - XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=10, eval_metric='logloss', verbosity=1)

# Train the model
xgb_model.fit(X_train, Y_train)

# Model Evaluation
X_train_prediction = xgb_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Training Accuracy:', training_data_accuracy)

X_test_prediction = xgb_model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Test Accuracy:', test_data_accuracy)

# Building a Predictive System
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
prediction = xgb_model.predict(input_data_as_numpy_array)

if prediction[0] == 0:
    print("\033[1mThe Person does not have Parkinson's Disease\033[0m")
else:
    print("\033[1mThe Person has Parkinson's\033[0m")

# Saving the trained model
filename = 'parkinsons_xgboost_model.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

# Loading the saved model
loaded_model = pickle.load(open(filename, 'rb'))
