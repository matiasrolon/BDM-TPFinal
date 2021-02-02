# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 01:31:55 2021

@author: Matias Rolon
"""


import pandas as pd
import numpy as np
import re
import dateutil.parser
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error 
    
from sklearn import metrics
from random import random
import matplotlib.pyplot as plt
from sklearn import tree

chessGamesData = pd.read_csv('games.csv')
chess = chessGamesData

'''
winner_and_status = []
for i in chess.index:  
        winner_and_status.append(str(chess["winner"][i]) + ' - ' + str(chess["victory_status"][i])) 

chess['winner_and_status']= winner_and_status
'''

chess_features =  chess[['white_rating', 'black_rating','white_id','black_id',
                         'opening_ply','turns','victory_status']]
chess_target = chess.winner

# Numerizamos atributos categoricos si los hubiera. (En este caso, solo channelId)
le = preprocessing.LabelEncoder()
for column_name in chess_features.columns:
    if (chess_features[column_name].dtype == object):
        chess_features[column_name] = le.fit_transform(chess_features[column_name])
   

# normalizamos los datos
chess_features = StandardScaler().fit_transform(chess_features)
# Entrenamos el modelo
X_train, X_test, y_train, y_test = train_test_split(chess_features, chess_target,test_size = 0.25, random_state=0)

# CLASIFICACION MEDIANTE REGRESION LOGISTICA
algReg = LogisticRegression()
algReg.fit(X_train,y_train)
y_pred = algReg.predict(X_test)
cnf_matriz = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matriz)
print("Accuracy regresión", metrics.accuracy_score(y_test,y_pred))

# CLASIFICACION MEDIANTE ARBOLES
arbol = tree.DecisionTreeClassifier(criterion='entropy')
arbol = arbol.fit(X_train, y_train)
y_pred = arbol.predict(X_test)
cnf_matriz = metrics.confusion_matrix(y_test,y_pred)
print(cnf_matriz)
print("Accuracy arbol:",metrics.accuracy_score(y_test, y_pred))

