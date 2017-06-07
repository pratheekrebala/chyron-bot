import cv2
import math
import numpy as np
from PIL import Image
import pytesseract
import string
import json
import streamlink
from tinydb import TinyDB, Query
from datetime import datetime, timedelta
import time
import tweepy
from tempfile import mkstemp
import os
import re

db = TinyDB('./chyron_db.json')

kellyanne_interview = 'https://www.youtube.com/watch?v=-otxWE6dBxk'
cnn_lots_of_videos = 'https://www.youtube.com/watch?v=YklGnu5NoE0'

livestream = ''
streams = streamlink.streams(livestream)

with open('credentials.json') as data_file:    
    creds = json.load(data_file)

auth = tweepy.OAuthHandler(creds['consumer_key'], creds['consumer_secret'])
auth.set_access_token(creds['access_key'], creds['access_secret'])

api = tweepy.API(auth)

videoFile = "test.mp4"
vidcap = cv2.VideoCapture(streams['720p'].url)
success,image = vidcap.read()

# Get Video every five seconds (assume 30 fps)
seconds = 5
fps = 30 #vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

allchars = string.maketrans('', '')
nonletter = allchars.translate(allchars, string.letters)

#Extract and Mask Chyron every 5 seconds.

dimensions = None

chyron_table = db.table('chyrons')
instance_table = db.table('instances')

def getCurrentTimestamp():
    return time.mktime(datetime.now().timetuple())

def createChyron(text):
    current_chyron = {'text': text, 'first_seen': getCurrentTimestamp(), 'tweeted': False, 'tweeted_at': None}
    return chyron_table.insert(current_chyron)

def searchOrCreateChyron(text):
    Chyron = Query()
    chyrons = chyron_table.search(Chyron.text == text)
    if not chyrons:
        #Chyron doesn't exist creating now.
        return createChyron(text)
    else: return chyrons[0].eid

def updateDB(text, image):
    chyron_id = searchOrCreateChyron(text)
    current_instance = {'chyron_id':chyron_id, 'seen': getCurrentTimestamp()}
    instance_table.insert(current_instance)
    shouldTweet(chyron_id, image)

def shouldTweet(chyron_id, image):
    Chyron = Query()
    current_chyron = chyron_table.get(eid=chyron_id)
    last_24_hrs = time.mktime((datetime.now() - timedelta(days=1)).timetuple())
    last_3_mins = time.mktime((datetime.now() - timedelta(minutes=3)).timetuple())
    instances = instance_table.search((Chyron.chyron_id==chyron_id) & (Chyron.seen > last_3_mins)) # Get all instances in last 24 hours.
    if len(instances) > 5 and (current_chyron["tweeted"] == False or current_chyron['tweeted_at'] < last_24_hrs):
        sendTweet(current_chyron, image)
        current_chyron['tweeted'] = True
        current_chyron['tweeted_at'] = getCurrentTimestamp()
        chyron_table.update(current_chyron, eids=[chyron_id])

def sendTweet(current_chyron, image):
    print('Tweeting: ' + current_chyron['text'])
    temp_path = './temp.jpg'
    cv2.imwrite(temp_path, image)
    print(temp_path)
    print(api.update_with_media(filename=temp_path, status=current_chyron['text']))

commercial_airing_for = 0

while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = vidcap.read()
    if frameId == 1:
        dimensions = dict(zip(('height', 'width', 'channels'), image.shape)) 
        lowerThirdStart = dimensions['height'] - int(dimensions['height'] * 0.21)
        lowerThirdEnd = dimensions['height'] - int(dimensions['height'] * 0.09)
        lowerThirdRight = dimensions['width'] - int(dimensions['width'] * 0.15)
        lowerThirdLeft = int(dimensions['width'] * 0.02)

        commercialStart = dimensions['height'] - int(dimensions['height'] * 0.11)
        commercialEnd = dimensions['height'] - int(dimensions['height'] * 0.075)
        commercialRight = dimensions['width'] - int(dimensions['width'] * 0.043)
        commercialLeft = dimensions['width'] - int(dimensions['width'] * 0.14)
    if frameId % (fps * 1) == 0:
        commercialDetect = image[commercialStart:commercialEnd,commercialLeft:commercialRight]
        iscommercial = pytesseract.image_to_string(Image.fromarray(commercialDetect)).lower()

        if 'am' in iscommercial or 'pm' in iscommercial:
            commercial_airing_for = 0
        else: commercial_airing_for += 1
        # cv2.imshow('frame', image)

    if frameId % multiplier == 0:
        lowerThird = image[lowerThirdStart:lowerThirdEnd,lowerThirdLeft:lowerThirdRight]
        text = pytesseract.image_to_string(Image.fromarray(lowerThird))
        text = text.encode('ascii',errors='ignore')
        text = re.sub(r'(?![A-Z])0', 'O', text)
        if len(text) > 4:
            #Text has to be capitalized and show hasn't aired for 10 seconds (based on timestamp in corner of the screen.)
            if text.isupper() and commercial_airing_for < 10:
                text = text.replace('0N', 'ON').replace('\n', ' ')
                print('CHYRON: ' + text)
            elif commercial_airing_for > 10:
                print('Commercial has been airing for: ' + str(commercial_airing_for) + ' seconds.')
                #updateDB(text, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()