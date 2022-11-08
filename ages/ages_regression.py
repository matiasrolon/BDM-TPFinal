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

# VARIABLES GLOBALES
PATH_ROOT = './'
PATH_DATA = './data/AgeDataset-V1.csv'

def tokenizer(text):
    text = re.sub('[-\[!\"\$%&*\(\)=/|:,]',' ',text)    # Quita simbolos especiales
    text = re.sub('[ ]+',' ',text)                      # Reemplaza los espacios en blanco repetidos por uno solo
    result = text.split(' ')                            # Separa las palabras resultantes en un array.
    return result


if __name__ == "__main__":
    data = pd.read_csv(PATH_DATA)
    # Selecciono las primeras 5000 filas para un procesamiento mas rapido en la fase de prueba.
    data = data.iloc[1:50]
    print(data)

    # TODO: preprocesamiento de los datos. Reemplazar o descartar registros con valores nulos, atípicos, ruido, etc.

    # Inicializo arrays de nuevos features.
    # Features calculados con el dataset
    birthCenturyArr = []
    birthDecadeArr = []
    deathCenturyArr = []
    deathDecadeArr = []
    qantOccupationsArr = []
    qantCountriesArr = []
    qantFeaturedEventsArr = []

    # Calculo los features para cada instancia
    for i in data.index:
        # Datos originales
        birthYear = data['Birth year'][i]
        deathYear = data['Death year'][i]
        occupation = str(data['Occupation'][i])
        country = str(data['Country'][i])

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
        # Calcula cantidad de nacionalidades
        qCountries = len(country.split(";")) if country.strip != "" else 0
        # TODO: Calcula eventos destacados en la vida
        qEvents = 0

        # Agrega nuevos datos a sus respectivos arrays
        birthCenturyArr.append(birthCentury)
        deathCenturyArr.append(deathCentury)
        birthDecadeArr.append(birthDecade)
        deathDecadeArr.append(deathDecade)
        qantOccupationsArr.append(qOccupations)
        qantCountriesArr.append(qCountries)
        qantFeaturedEventsArr(qEvents)

    # print(qantCountriesArr)

    # TODO: normalización de los datos.




