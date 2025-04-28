# %% [markdown]
# Importing the Dependencies

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import pickle

# %% [markdown]
# Data Collection & Analysis

# %%
# loading the data from csv file to a Pandas DataFrame
parkinsons_data = pd.read_csv('Parkinson\parkinsons.csv')

# %%
# printing the first 5 rows of the dataframe
parkinsons_data.head()

# %%
# number of rows and columns in the dataframe
parkinsons_data.shape

# %%
# getting more information about the dataset
parkinsons_data.info()

# %%
# checking for missing values in each column
parkinsons_data.isnull().sum()

# %%
# getting some statistical measures about the data
parkinsons_data.describe()

# %%
# distribution of target Variable
parkinsons_data['status'].value_counts()

# %%
# 1  --> Parkinson's Positive
# 
# 0 --> Healthy

# %%
parkinsons_data['status'].unique()

# %%
parkinsons_data.dtypes

# %%
# grouping the data based on the target variable
numeric_columns = parkinsons_data.select_dtypes(include='float64').columns
numeric_columns_and_status = numeric_columns.append(pd.Index(['status']))
grouped_data = parkinsons_data[numeric_columns_and_status].groupby('status').mean()

# Display the result
print(grouped_data)

# %% [markdown]
# Data Pre-Processing

# %% [markdown]
# Separating the features & Target

# %%
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']

# %%
print(X)

# %%
print(Y)

# %% [markdown]
# Splitting the data into training and test data

# %%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# %%
print(X.shape, X_train.shape, X_test.shape)

# %% [markdown]
# Model Training

# %% [markdown]
# CatBoost Classifier Model

# %%
catboost_model = CatBoostClassifier(iterations=1000, depth=10, learning_rate=0.1, loss_function='Logloss', verbose=200)

# %% 
# training the CatBoost model with training data
catboost_model.fit(X_train, Y_train)

# %% [markdown]
# Model Evaluation

# %% [markdown]
# Accuracy Score

# %% 
# accuracy score on training data
X_train_prediction = catboost_model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy score of training data : ', training_data_accuracy)

# %% 
# accuracy score on test data
X_test_prediction = catboost_model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy score of test data : ', test_data_accuracy)

# %% [markdown]
# Building a Predictive System

# %% 
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# make prediction
prediction = catboost_model.predict(input_data_reshaped)

print(prediction)

if prediction[0] == 0:
    print("\033[1mThe Person does not have Parkinson's Disease\033[0m")
else:
    print("\033[1mThe Person has Parkinson's\033[0m")
pickle.dump(catboost_model, open('catboost_parkinsons_model.sav', 'wb'))
