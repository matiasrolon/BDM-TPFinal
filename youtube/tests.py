# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:46:06 2021

@author: rolon
"""

import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

#VARIABLES GLOBALES
DEVELOPER_KEY = 'AIzaSyDIh-eX0MqOgZyAQ_NMtNvVx7tsyujnbYM'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
PATH_STORAGED_DATA =  "./data/youtube-storage.txt"

youtubeAPI = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
channels_response = youtubeAPI.channels().list(part="snippet,statistics", id="UCvZ0P9-EmGz6SFUuFiZPXKw").execute()
fecha  = channels_response['items'][0]['snippet']['publishedAt'][0:10]

today = datetime.date.today()
someday = datetime.date(int(fecha[0:4]), int(fecha[5:7]), int(fecha[8:10]))
diff =  today - someday
print(channels_response['items'][0])
print('dias: ', diff.days)
print(channels_response['items'][0]['statistics']['subscriberCount'])
# tener en cuenta si tiene hiddenSubscriberCount en False o True.
#print(channels_response)
#UCWn_6CsjxM-VVAUHzKfKtGw

"""
file = open(PATH_STORAGED_DATA,"r")
storage = {}
for i in file.readlines()[1:]:
    i = i[:-1]
    values_line = i.split(',')
    print(values_line)    
    if not storage.get(values_line[0]):
        storage[values_line[0]] = {'duration_minutes': int(values_line[1]),
                                   'captions': bool(values_line[2]),
                                   'definition': values_line[3],
                                   'made_for_kids': bool(values_line[4]),
                                   'q_subscribers': int(values_line[5]),
                                   'channel_antiquity': int(values_line[6])
                                }    

print('='*30)
print(storage.get('GTp-0S82guE'))"""







