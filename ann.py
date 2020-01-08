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

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def built_classifier(optimizer): #builts ann classifier
    classifier = Sequential()                   
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))                                                 
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid', input_dim = 11 ))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn = built_classifier)

parameters = {"batch_size": [10, 25, 32],
              'nb_epoch': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search  = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_