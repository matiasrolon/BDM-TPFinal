# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:46:07 2021

@author: Matias Rolon
"""
import pandas as pd
import numpy as np
import re
    
import matplotlib.pyplot as plt

if __name__ == "__main__":
    youtubeData = pd.read_csv('US_youtube_trending_data.csv')
    yt = youtubeData
    print('Minimo numero de visualizaciones: ', min(yt['view_count']))
    print('Maximo numero de visualizaciones: ', max(yt['view_count']))

    # GRAFICOS 
    # de 1 a 10 millones, bins cada 1 millon.
    '''
    intervalos = list(range(1,80000000,1000000))
    plt.title('views')
    plt.xticks(intervalos, ["1m","","","","","","","","","1m",
                            "","","","","","","","","2m",
                            "","","","","","","","","3m",
                            "","","","","","","","","4m",
                            "","","","","","","","","5m",
                            "","","","","","","","","6m",
                            "","","","","","","","","7m",
                            "","","","","","","","","8m",
                            "","","","","","","","","9m",
                            "","","","","","","","","10m",])
    plt.hist(yt['view_count'],bins=intervalos)
    plt.grid(True)
    plt.show()
    '''
    # de 1 a 10 millones, bins cada 1 millon.
    '''
    intervalos = list(range(1,10000000,100000))
    plt.title('views')
    plt.xticks(intervalos, ["1m","","","","","","","","","1m",
                            "","","","","","","","","2m",
                            "","","","","","","","","3m",
                            "","","","","","","","","4m",
                            "","","","","","","","","5m",
                            "","","","","","","","","6m",
                            "","","","","","","","","7m",
                            "","","","","","","","","8m",
                            "","","","","","","","","9m",
                            "","","","","","","","","10m",])
    plt.hist(yt['view_count'],bins=intervalos)
    plt.grid(True)
    plt.show()
    '''
    
    
    print(max(yt['view_count']))
    intervalos = list(range(1,3000000,100000)) + list(range(3000001,10000000,1000000)) + list(range(10000000,240000000,225000000))
    #intervalos = list(range(1,10000000,100000))
    plt.title('views')
    '''
    plt.xticks(intervalos, ["1m","","","","","","","","","1m",
                            "","","","","","","","","2m",
                            "","","","","","","","","3m",
                            "","","","","","","","","4m",
                            "","","","","","","","","5m",
                            "","","","","","","","","6m",
                            "","","","","","","","","7m",
                            "","","","","","","","","8m",
                            "","","","","","","","","9m",
                            "","","","","","","","","10m",])
    '''
    plt.hist(yt['view_count'],bins=intervalos)
    plt.grid(True)
    plt.show()

    
    '''
     Hasta los 3 millones, predecir de a 100 mil
     De 3 millones a 10 millones, predecir de a 1 millon
    '''