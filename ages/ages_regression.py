# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:44:15 2021

@author: Matias Rolon
"""

import pandas as pd
import math
import numpy as np
import re
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# VARIABLES GLOBALES
PATH_ROOT = './'
PATH_DATA = './data/AgeDataset-V1.csv'
TARGET_COLUMN = "Age of death"

def tokenizer(text):
    text = re.sub('[-\[!\"\$%&*\(\)=/|:,]',' ',text)    # Quita simbolos especiales
    text = re.sub('[ ]+',' ',text)                      # Reemplaza los espacios en blanco repetidos por uno solo
    result = text.split(' ')                            # Separa las palabras resultantes en un array.
    return result


if __name__ == "__main__":
    data = pd.read_csv(PATH_DATA)
    # Selecciono las primeras 5000 filas para un procesamiento mas rapido en la fase de prueba.
    data = data.iloc[1:5000]
    print(data)

    # PREPROCESAMIENTO DE LOS DATOS ==================================================================================================
    # TODO: preprocesamiento de los datos. Reemplazar o descartar registros con valores nulos, atípicos, ruido, etc.
    data.loc[data['Manner of death'].isnull(), 'Manner of death'] = 'Unknown'
    data.loc[data['Death year'].isnull(), 'Death year'] = 0
    data.loc[data['Birth year'].isnull(), 'Birth year'] = 0

    # AGREGA NUEVOS FEATURES CALCULADOS AL DATASET ===================================================================================
    # Inicializo arrays de nuevos features
    # Features calculados con el dataset
    birthCenturyArr = []
    birthDecadeArr = []
    deathCenturyArr = []
    deathDecadeArr = []
    qantOccupationsArr = []
    qantCountriesArr = []
    qantFeaturedEventsArr = []

    # auxiliares
    qMaxOcu = 0

    # Calculo los features para cada instancia
    for i in data.index:
        # Datos originales
        birthYear = data['Birth year'][i]
        deathYear = data['Death year'][i]
        occupation = str(data['Occupation'][i])
        country = str(data['Country'][i])
        shortDescription = str(data['Short description'][i])

        # Calcula siglo de nacimiento
        birthCentury = int((int(birthYear)/100)-1)
        # Calcila siglo de muerte
        deathCentury = int((int(deathYear)/100)-1)
        # Calcula decada de nacimiento
        if birthYear < 0:
            birthYear = int(birthYear) * (-1)
        birthDecade = math.floor((birthYear - 1) % 100 / 10) * 10
        # Calcula decada de muerte
        if deathYear < 0:
            deathYear = int(deathYear) * (-1)
        deathDecade = math.floor((deathYear - 1) % 100 / 10) * 10
        # Calcula cantidad de ocupaciones
        qOccupations = len(occupation.split(";")) if occupation.strip != "" else 0
        if qOccupations > qMaxOcu:
            qMaxOcu = qOccupations
        # Calcula cantidad de nacionalidades
        qCountries = len(country.split(";")) if country.strip != "" else 0
        # Calcula eventos destacados de su vida
        qEvents = 0
        if shortDescription is not None and shortDescription.strip!="":
            splitComma = shortDescription.split(",")
            if len(splitComma) > 0:
                qEvents = len(splitComma)
            else:
                splitPointComma = shortDescription.split(";")
                if len(splitPointComma) > 0:
                    qEvents = len(splitPointComma)
                else:
                    qEvents = 1
        else:
            qEvents = 0

        if qEvents == 0 and "and " in shortDescription:
            qEvents += 1

        # Agrega nuevos datos a sus respectivos arrays
        birthCenturyArr.append(birthCentury)
        deathCenturyArr.append(deathCentury)
        birthDecadeArr.append(birthDecade)
        deathDecadeArr.append(deathDecade)
        qantOccupationsArr.append(qOccupations)
        qantCountriesArr.append(qCountries)
        qantFeaturedEventsArr.append(qEvents)

    # print(qantCountriesArr)
    data['birthCentury'] = birthCenturyArr
    data['deathCentury'] = deathCenturyArr
    data['birthDecade'] = birthDecadeArr
    data['deathDecade'] = deathDecadeArr
    data['qantOccupations'] = qantOccupationsArr
    data['qantCountries'] = qantCountriesArr
    data['qantFeaturedEvents'] = qantFeaturedEventsArr

    # Numerizamos atributos categoricos si los hubiera.
    le = preprocessing.LabelEncoder()
    for column_name in data.columns:
        if (data[column_name].dtype == object) and (column_name != TARGET_COLUMN):
            data[column_name] = le.fit_transform(data[column_name])

    print("maxima cantidad de ocupacioens encontradas en un solo registro:",str(qMaxOcu))
    print(data["Occupation"])
    # Separamos entre columnas features y columna target
    data_features = data[['Gender', 'Country', 'Occupation', 'Birth year', 'Death year',
        'Manner of death', 'birthCentury', 'deathCentury', 'birthDecade', 'deathDecade', 'qantOccupations',
        'qantCountries', 'qantFeaturedEvents']]

    data_target = data[TARGET_COLUMN]

    # normalizamos los datos
    data_features = StandardScaler().fit_transform(data_features)
    # print(data_features)
    # TODO: normalización de los datos.




