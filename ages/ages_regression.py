# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:44:15 2021

@author: Matias Rolon
"""
import copy

import pandas as pd
import math
import numpy as np
import re
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# VARIABLES GLOBALES
PATH_ROOT = './'
PATH_DATA = './data/AgeDataset-V1.csv'
TARGET_COLUMN = "Age of death"
Q_REGISTERS = 500

def tokenizer(text):
    text = re.sub('[-\[!\"\$%&*\(\)=/|:,]',' ',text)    # Quita simbolos especiales
    text = re.sub('[ ]+',' ',text)                      # Reemplaza los espacios en blanco repetidos por uno solo
    result = text.split(' ')                            # Separa las palabras resultantes en un array.
    return result

def normalizateCategoricalColumn(value, idx, newColumns):
    if value is not None:
        for valueName in value.split(";"):
            if valueName in newColumns:
                # print("es ", occName ," index:", idx)
                newColumns[valueName.strip()][idx] = 1
            else:
                # print("creo", occName, " index", idx)
                newColumns[valueName.strip()] = np.zeros(dataSize)
                # print("largo nueva column", len(newColumnsOccupation[occName.strip()]))
                newColumns[valueName.strip()][idx] = 1
    else:
        newColumns["unknown"][idx] = 1

    return newColumns

if __name__ == "__main__":
    data = pd.read_csv(PATH_DATA)
    # Selecciono las primeras N filas para un procesamiento mas rapido en la fase de prueba.
    if Q_REGISTERS is not None:
        data = data.iloc[0:Q_REGISTERS]

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
    dataSize = len(data)

    defaultNewColumnsObject = {
        "unknown": np.zeros(dataSize)
    }
    newColumnsOccupation = copy.deepcopy(defaultNewColumnsObject)
    newColumnsCountry = copy.deepcopy(defaultNewColumnsObject)
    newColumnsMannerDeath = copy.deepcopy(defaultNewColumnsObject)

    print("largo dataset",dataSize)
    # Calculo los features para cada instancia
    for i in data.index:
        # Datos originales
        birthYear = data['Birth year'][i]
        deathYear = data['Death year'][i]
        occupation = str(data['Occupation'][i])
        country = str(data['Country'][i])
        shortDescription = str(data['Short description'][i])
        mannerOfDeath = str(data['Manner of death'][i])

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

        # Verifica ocupaciones y paises nuevos, para crear columnas binarias segun los valores de todos los registros.
        # Se utiliza esta logica custom, ya que con get_dummies no se logra destinguir cuando un registro tiene varias ocupaciones separadas por ";"
        idx = i
        newColumnsOccupation = normalizateCategoricalColumn(occupation, idx, newColumnsOccupation)
        newColumnsCountry = normalizateCategoricalColumn(country, idx, newColumnsCountry)
        newColumnsMannerDeath = normalizateCategoricalColumn(mannerOfDeath, idx, newColumnsMannerDeath)

    print("nuevas profesioens encontradas", newColumnsOccupation)
    # print(qantCountriesArr)
    data['birthCentury'] = birthCenturyArr
    data['deathCentury'] = deathCenturyArr
    data['birthDecade'] = birthDecadeArr
    data['deathDecade'] = deathDecadeArr
    data['qantOccupations'] = qantOccupationsArr
    data['qantCountries'] = qantCountriesArr
    data['qantFeaturedEvents'] = qantFeaturedEventsArr

    newColumnsArray = [
        [newColumnsOccupation, "occupation"],
        [newColumnsCountry, "country"],
        [newColumnsMannerDeath, "manner_death"]
    ]
    for newColumnsObject in newColumnsArray:
        print(newColumnsObject[0].keys())
        for column in newColumnsObject[0].keys():
            newName = newColumnsObject[1] + "_" + column.replace(" ", "_").lower()
            #data[newName] = newColumnsObject[0][column]
            data.insert(1, newName, newColumnsObject[0][column])

    data_target = data[TARGET_COLUMN]
    # Eliminamos columna que no nos serviran para el modelo de regresión o que ya vueron normalizadas
    data_features = data.drop([TARGET_COLUMN, 'Id', 'Name', 'Short description', 'Occupation', 'Country', "Manner of death"], axis=1)
    print("columnas features", data_features.columns)

    # Numerizamos atributos categoricos si los hubiera.
    le = preprocessing.LabelEncoder()
    for column_name in data_features.columns:
        if data_features[column_name].dtype == object:
            data_features[column_name] = le.fit_transform(data_features[column_name])

    # Normalizamos los datos
    data_features = StandardScaler().fit_transform(data_features)

    #print(data_features)





