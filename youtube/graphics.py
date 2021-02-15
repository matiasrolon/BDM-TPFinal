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
    youtubeData = pd.read_csv('./data/US_youtube_trending_data.csv')
    yt = youtubeData
    print('Min de visualizaciones: ', min(yt['view_count']))
    print('Max de visualizaciones: ', max(yt['view_count']))

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

    intervalos = list(range(1,10000000,100000))
    plt.title('views')
    plt.xticks(intervalos, ["","","","","","","","","","1m",
                            "","","","","","","","","","2m",
                            "","","","","","","","","","3m",
                            "","","","","","","","","","4m",
                            "","","","","","","","","","5m",
                            "","","","","","","","","","6m",
                            "","","","","","","","","","7m",
                            "","","","","","","","","","8m",
                            "","","","","","","","","","9m",
                            "","","","","","","","","","10m"])
    plt.hist(yt['view_count'],bins=intervalos)
    plt.grid(True)
    plt.show()

    '''
    print(max(yt['view_count']))
    intervalos = list(range(1,3000000,100000)) + list(range(3000001,10000000,1000000)) + list(range(10000000,240000000,225000000))
    #intervalos = list(range(1,10000000,100000))
    plt.title('views')
    plt.hist(yt['view_count'],bins=intervalos)
    plt.grid(True)
    plt.show()
    '''
    
    
    '''
    
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

    # Clasifico la instancia
    # viewsRanges.append(setRange(yt["view_count"][i]))
   '''