# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:54:28 2019

@author: tarun
"""

# ---------------------- Data preprocessing ---------------------- 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13 ].values    
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])              # X_1 is for country

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])              # X_2 is for male/female

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ---------------------- ANN ---------------------- 

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout       # To avoid overfitting by disabling some neurons

# Intializing the ANN
classifier = Sequential()                   
        #   Hidden Layer 1
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))     
classifier.add(Dropout(p = 0.1))    #p = fraction of neurons to disablie (0 to 1, gerally less than 0.5 to avoid underfitting)
        # Hidden Layer 2                                                                                                
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))
        # Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 11 ))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
                                                                         
# ---------------------- Making prediction & evaluating model---------------------- 
                                                                         
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)               # accuracy  = (1564 +115 ) /2000 = 0.8395 
