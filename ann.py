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
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])              # X_1 is for country

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])              # X_2 is for male/female

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
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

                    # Intializing the ANN
classifier = Sequential()                   
    #   Hidden Layer 1
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))     
    # Hidden Layer 2                                                                                                
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    # Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 11 ))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
                                                                         
# ---------------------- Making prediction & evaluating model---------------------- 
                                                                         
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)               # accuracy  = (1556 +131 ) /2000 = 0.8435
