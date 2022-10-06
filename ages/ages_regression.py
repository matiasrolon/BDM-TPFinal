# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:44:15 2021

@author: Matias Rolon
"""

import pandas as pd
import numpy as np
import re
import dateutil.parser
import matplotlib.pyplot as plt
import datetime

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error 
from random import random
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# VARIABLES GLOBALES
DEVELOPER_KEY = 'AIzaSyDIh-eX0MqOgZyAQ_NMtNvVx7tsyujnbYM'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
PATH_STORAGED_DATA = "./data/youtube-storage.txt"

def tokenizer(text):
    text = re.sub('[-\[!\"\$%&*\(\)=/|:,]',' ',text)    # Quita simbolos especiales
    text = re.sub('[ ]+',' ',text)                      # Reemplaza los espacios en blanco repetidos por uno solo
    result = text.split(' ')                            # Separa las palabras resultantes en un array.
    return result


if __name__ == "__main__":
    # Primero cargo en memoria los datos que ya tengo de la API.
    print("[+] Cargando en memoria (dict) los datos ya obtenidos de la API...")
    file = open(PATH_STORAGED_DATA,"r")
    storaged_data = {}
    for i in file.readlines()[1:]:
        print("Se cargo un video desde el archivo")
        i = i[:-1]                  # quito el /n
        values_line = i.split(',')  # separo los valores de una fila.
        if not storaged_data.get(values_line[0]):   # Busco por video_id si no esta guardado ya
            storaged_data[values_line[0]] = {'duration_minutes': int(values_line[1]),
                                       'caption': bool(values_line[2]),
                                       'definition': values_line[3],
                                       'made_for_kids': bool(values_line[4]),
                                       'q_subscribers': int(values_line[5]),
                                       'channel_antiquity': int(values_line[6])
                                    }    
    file.close()
    print("[+] Diccionario generado")
    
    file = open(PATH_STORAGED_DATA,"a+")
    #file.write('video_id, duration_minutes, caption, definition, made_for_kids, q_subscribers, channel_antiquity'+ '\n')
    youtubeAPI = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    youtubeData = pd.read_csv('./data/US_youtube_trending_data.csv')
    yt = youtubeData
    # Elimino 29000 filas para un procesamiento mas rapido en la fase de prueba.
    yt = youtubeData.iloc[1:7300]
    # Inicializo arrays de nuevos features.
    # Features calculados con el dataset
    lengthTitle = []
    upperWords = []
    questionMarks = []
    ingeniousTags = []
    lengthDescrip = []
    publishedHour = []
    viewsRanges = []
    # features pedidos a la API
    durations = []
    captions = []
    definitions = []
    madeForKids = []
    dateChannel = []
    subsCount = []
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
        questionMarks.append(title.count('?'))
        
        # Cant de tags 
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
       
        # Request a Youtube API v3 apartir del video_id
        # Primero verifico si antes no esta grabada en disco la info de ese video.  
        if not storaged_data.get(yt["video_id"][i]):
            print("no esta en diccionario... busco en api")
            resultRequest = youtubeAPI.videos().list( part='contentDetails,status',id=yt["video_id"][i]).execute()        
            if i % 10 == 0: print('Solicitud nº',i)
            if len(resultRequest['items'])>0:
                channelRequest = youtubeAPI.channels().list(part="snippet,statistics", id=yt["channelId"][i]).execute()
                
                video = resultRequest['items'][0]
                durationStr = video['contentDetails']['duration']
                try:
                    durationValue = int(durationStr[2:durationStr.index('M')])
                except ValueError:
                    # dura menos de un minuto
                    durationValue = 0                    
                definitionValue=video['contentDetails']['definition']
                captionValue = bool(video['contentDetails']['caption'])
                mfkValue = bool(video['status']['madeForKids'])    
                if channelRequest['items'][0]['statistics']['hiddenSubscriberCount']:
                    subsCountValue = 0
                else: subsCountValue = int(channelRequest['items'][0]['statistics']['subscriberCount'])
                
                dateReq = channelRequest['items'][0]['snippet']['publishedAt'][0:10]
                today = datetime.date.today()
                dateInitChannel = datetime.date(int(dateReq[0:4]), int(dateReq[5:7]), int(dateReq[8:10]))
                diff =  today - dateInitChannel 
                dateChannelValue = diff.days    
            else:
                print("Error, no existen datos del video ", yt["video_id"][i])
                durationValue = durations[len(durations)-1]    
                definitionValue= definitions[len(definitions)-1]
                captionValue = captions[len(captions)-1]
                mfkValue = madeForKids[len(madeForKids)-1]
                dateChannelValue = 0   
                subsCountValue = 0
                
            file.write( yt["video_id"][i] + ',' +
                        str(durationValue) + ',' + 
                        str(captionValue) + ',' +
                        str(definitionValue) + ',' +
                        str(mfkValue) + ',' +
                        str(subsCountValue) + ',' +
                        str(dateChannelValue) + '\n'
            )
        else:   # Si ya estaba en disco, se cargo en memoria en el primer paso del codigo, entonces busco en el dict.
            durationValue = storaged_data.get(yt["video_id"][i])['duration_minutes']
            definitionValue= storaged_data.get(yt["video_id"][i])['definition'] 
            captionValue = storaged_data.get(yt["video_id"][i])['caption']
            mfkValue = storaged_data.get(yt["video_id"][i])['made_for_kids']
            dateChannelValue = storaged_data.get(yt["video_id"][i])['channel_antiquity']
            subsCountValue = storaged_data.get(yt["video_id"][i])['q_subscribers']
            
        durations.append(durationValue)                
        definitions.append(definitionValue)
        captions.append(captionValue)
        madeForKids.append(mfkValue)
        dateChannel.append(dateChannelValue)
        subsCount.append(subsCountValue)
     
    file.close()
    # Agregamos los nuevos features al dataset original
    yt['length_title'] = lengthTitle
    yt['q_upper_words'] = upperWords
    yt['q_question_marks'] = questionMarks
    yt['q_tags'] = ingeniousTags
    yt['length_description'] = lengthDescrip
    yt['published_hour'] = publishedHour

    yt['duration_minutes'] = durations
    yt['caption'] = captions
    yt['definition'] = definitions
    yt['made_for_kids'] = madeForKids
    yt['channel_antiquity'] = dateChannel
    yt['q_subscribers'] = subsCount       
    # Eliminamos las columnas que no vayamos a utilizar 
    # Definimos los features y el target.
    
    yt_features = yt[['length_title','q_upper_words','q_question_marks','q_tags','length_description',
                      'published_hour','channelId','categoryId','ratings_disabled','comments_disabled',
                      'duration_minutes','caption','definition','made_for_kids',
                      'channel_antiquity', 'q_subscribers',
                      #'comment_count','likes','dislikes'
                      ]]
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
    print('*'*32)  
    #result =  modelLinear.predict([23,False,False,5,0,0,22,19,])

