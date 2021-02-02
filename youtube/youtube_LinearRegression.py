# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:44:15 2021

@author: Matias Rolon
"""

import pandas as pd
import numpy as np
import re
import dateutil.parser
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error 
    
from random import random
import matplotlib.pyplot as plt


def tokenizer(text):
    text = re.sub('[-\[!\"\$%&*\(\)=/|:,]',' ',text)    # Quita simbolos especiales
    text = re.sub('[ ]+',' ',text)                      # Reemplaza los espacios en blanco repetidos por uno solo
    result = text.split(' ')                            # Separa las palabras resultantes en un array.
    return result


def setRange(num):
    result = 0
    if num<100000: result=0
    elif num<200001: result=1
    elif num<300001: result=2
    elif num<400001: result=3
    elif num<500001: result=4
    elif num<600001: result=5
    elif num<700001: result=6
    elif num<800001: result=7
    elif num<900001: result=8
    elif num<1000001: result=9
    elif num<1200001: result=10
    elif num<1300001: result=11
    elif num<1400001: result=12
    elif num<1500001: result=13
    elif num<1600001: result=14
    elif num<1700001: result=15
    elif num<1800001: result=16
    elif num<1900001: result=17
    elif num<2000001: result=18
    elif num<2100001: result=19
    elif num<2200001: result=20
    elif num<2300001: result=21
    elif num<2400001: result=22
    elif num<2500001: result=23
    elif num<2600001: result=24
    elif num<2700001: result=25
    elif num<2800001: result=26
    elif num<2900001: result=27
    elif num<3000001: result=28 
    elif num<4000001: result=29
    elif num<5000001: result=30
    elif num<6000001: result=31  
    elif num<7000001: result=32
    elif num<8000001: result=33
    elif num<9000001: result=34
    elif num<10000001: result=35
    else: result=36
    return result


if __name__ == "__main__":
    youtubeData = pd.read_csv('US_youtube_trending_data.csv')
    yt = youtubeData
    # Elimino 29000 filas para un procesamiento mas rapido en la fase de prueba.
    #yt = youtubeData.iloc[1:1000]
    
    # Inicializo arrays de nuevos features.
    lengthTitle = []
    upperWords = []
    #questionMarks = []
    ingeniousTags = []
    lengthDescrip = []
    publishedHour = []
    viewsRanges = []
    # Calculo los features para cada instancia
    for i in yt.index:  
        title =  yt["title"][i]
        # Cant palabras del titulo
        wordsTitle = tokenizer(title)
        # Cant palabras en mayuscula del titulo
        lengthTitle.append(len(wordsTitle))     
        q_upper = 0
        for word in wordsTitle: 
           if word.isupper(): q_upper+=1  
        upperWords.append(q_upper) 
        # Cant de signos de preguntas del titulo
        # questionMarks.append(title.count('?'))
        # Cant de tags que no se encuentran en el titulo video
        tagsArray = yt["tags"][i].split('|')
        q_tags = 0
        if tagsArray[0]!='[None]': 
            q_tags = len(tagsArray)
        ingeniousTags.append(q_tags)
        # Cant de palabras de la descripcion
        descripArray = tokenizer(str(yt["description"][i]))
        if descripArray[0]=='': lengthDescrip.append(0)
        else: lengthDescrip.append(len(descripArray))
        # Hora del dia que fue subido el video. (Es un rango. Ejemplo: 08 = fue subido de 8:00 a 8:59)
        publishDate = dateutil.parser.parse(yt["publishedAt"][i])
        publishedHour.append( int(publishDate.strftime('%H')) )
        # Clasifico la instancia
        viewsRanges.append(setRange(yt["view_count"][i]))
        
    # Agregamos los nuevos features al dataset original
    yt['length_title'] = lengthTitle
    yt['q_upper_words'] = upperWords
    #yt['q_question_marks'] = questionMarks
    yt['q_tags'] = ingeniousTags
    yt['length_description'] = lengthDescrip
    yt['published_hour'] = publishedHour
    #yt['view_range'] = viewsRanges
    # Eliminamos las columnas que no vayamos a utilizar 
    # Definimos los features y el target.
    yt_features = yt.drop(['video_id', 'title','publishedAt','channelTitle','channelId','tags','thumbnail_link',
                  'description','trending_date','view_count','categoryId','comment_count',
                  'ratings_disabled','comments_disabled'], axis=1)
    yt_target = yt['view_count']
    
    # Numerizamos atributos categoricos si los hubiera. (En este caso, solo channelId)
    print('Features finales: ',yt_features.columns)
    le = preprocessing.LabelEncoder()
    for column_name in yt_features.columns:
        if (yt_features[column_name].dtype == object):
            yt_features[column_name] = le.fit_transform(yt_features[column_name])
    
    # normalizamos los datos
    yt_features = StandardScaler().fit_transform(yt_features)
    
    # Partimos el conjunto de entrenamiento. Para añadir replicabilidad usamos el random state
    X_train, X_test, y_train, y_test = train_test_split(yt_features, yt_target, test_size=0.2, random_state=2)
    print('Cantidad de datos de entrenamiento: ',X_train.shape)
    print('Cantidad de datos de testing:', X_test.shape)
    # Aplico los distintos modelos de regresion (lineal, lasso y ridge)
    #Lineal
    modelLinear = LinearRegression().fit(X_train,y_train)
    y_predict_linear = modelLinear.predict(X_test)
    #Lasso
    modelLasso = Lasso(alpha=0.2).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)
    #Ridge
    modelRidge = Ridge(alpha=0.2).fit(X_train, y_train)   
    y_predict_ridge = modelRidge.predict(X_test)
    
    #print('Linear score: ',regressor.score(X_test, y_test))
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print('*'*32)
    print('Linear loss: ', linear_loss)
    print('Lasso loss: ', lasso_loss)
    print('Ridge loss: ', ridge_loss)
    print('*'*32)
    print('Linear score: ', modelLinear.score(X_test,y_test))
    print('Lasso score: ', modelLasso.score(X_test,y_test))
    print('Ridge score: ', modelRidge.score(X_test,y_test))
        
    #result =  modelLinear.predict([23,False,False,5,0,0,22,19,])

