# -*- coding: utf-8 -*-
"""Project Heart Disease Prediction.ipynb

#Heart Disease Prediction

StepI: Importing the Library
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Step II:Data Collection and Processing"""

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Excel file\Desktop\Miiii\deploy\heart_disease_data.csv")

# print first 5 rows of the dataset
heart_data.head()

# print last 5 rows of the dataset
heart_data.tail()

# number of rows and columns in the dataset
heart_data.shape

# getting some info about the data
heart_data.info()

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of Target Variable
heart_data['target'].value_counts()

"""1 --> Defective Heart

0 --> Healthy Heart

Step III: Data Visulisation
"""

#Shows how many people have heart disease vs not.
heart_data['target'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()

#Helps show societal impact — how risk increases with age.
plt.scatter(heart_data['age'],heart_data['thalach'])
plt.title("Age vs Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.show()

#Chest pain Type distribution
heart_data['cp'].value_counts().plot(kind='bar')
plt.title("Chest Pain Type Distribution")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

#Cholesterol level
plt.hist(heart_data['chol'], bins=20)
plt.title("Cholesterol Level Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10,8))
plt.imshow(heart_data.corr(), interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(heart_data.columns)), heart_data.columns, rotation=90)
plt.yticks(range(len(heart_data.columns)), heart_data.columns)
plt.title("Feature Correlation Heatmap")
plt.show()

"""Step IV: Splitting the Features and Target"""

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

print(X)

print(Y)

"""Step V: Splitting the Data into Training data & Test Data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Step VI: Model Training

Logistic Regression
"""

model = LogisticRegression(max_iter=1000)

# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

"""Step VII: Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, X_test_prediction)

plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()

"""Step VIII:Building a Predictive System"""

input_data = (63,1,3,150,260,1,0,130,1,2.8,0,2,3)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

"""Saved Trained Model using Pickle Library."""

import pickle

with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully to 'logistic_regression_model.pkl'")



