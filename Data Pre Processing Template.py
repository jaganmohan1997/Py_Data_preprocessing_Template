# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:08:56 2019

@author: Jagan Mohan
"""

# Data Pre Processing

## Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1 ].values

## Missing Value Treatment

### Find Missing Values:- isnull function can be only used for python dataframes
X1 = dataset.iloc[:,:-1]
X1.isnull() #Will show all the Missing values in the dataset
X1.isnull().sum() #Incase the datset is big you can get the number f missing values in each column
X1.isnull().sum().sum() #Get the count of total missing values

### Imputing the missing values
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X[:,1:3] = imputer.fit_transform(X[:,1:3])

## Encoding Categorical Values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
onehotencoder_x = OneHotEncoder(categorical_features = [0])
X = onehotencoder_x.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

## Train and Test Split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1234)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# This ends the sample templates that we can often use for data preprocessing in python.







