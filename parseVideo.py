import os
import re
import json
import time
import string
from datetime import datetime, timedelta

import cv2
from PIL import Image

import streamlink
import pytesseract
from tinydb import TinyDB, Query
import tweepy

# Keeps track of airings and chyrons in a json file.
db = TinyDB('./chyron_db.json')

# Lookup Stream
with open('streams.json') as fp:
    stream_list = json.load(fp)
    streams = streamlink.streams(stream_list['live'])

# Twitter Credentials
with open('credentials.json') as fp:    
    credentials = json.load(fp)

# Setup Tweepy
auth = tweepy.OAuthHandler(credentials['consumer_key'], credentials['consumer_secret'])
auth.set_access_token(credentials['access_key'], credentials['access_secret'])

api = tweepy.API(auth)

# videoFile = "test.mp4"

# Load Stream and pass onto openCV
vidcap = cv2.VideoCapture(streams['720p'].url)
success,image = vidcap.read()

# Get Video every five seconds (assuming 30 fps)
seconds = 5
fps = 30 #vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second - still testing this.
multiplier = fps * seconds

# Try to handle bad output from tesseract.
allchars = string.maketrans('', '')
nonletter = allchars.translate(allchars, string.letters)

dimensions = None

# Setup Tables
chyron_table = db.table('chyrons')
instance_table = db.table('instances')

# Return epoch value from timestamp
def getCurrentTimestamp():
    return time.mktime(datetime.now().timetuple())

# Store new Chyrons
def createChyron(text):
    current_chyron = {'text': text, 'first_seen': getCurrentTimestamp(), 'tweeted': False, 'tweeted_at': None}
    return chyron_table.insert(current_chyron)

# Look if Chyrons' exist
def searchOrCreateChyron(text):
    Chyron = Query()
    chyrons = chyron_table.search(Chyron.text == text)
    if not chyrons:
        #Chyron doesn't exist creating now.
        return createChyron(text)
    else: return chyrons[0].eid

# Handle Chyron
def updateDB(text, image):
    chyron_id = searchOrCreateChyron(text)
    current_instance = {'chyron_id':chyron_id, 'seen': getCurrentTimestamp()}
    instance_table.insert(current_instance)
    shouldTweet(chyron_id, image)

# Tweets Chyron if the text has appeared for the first time in the last 24 hours
# and has appeared for more than 5 intervals (currently 30 seconds) in the last 3 minutes.
# Performs decently to handle bad output from tesseract due to text transitions

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

# Store the image in a temporary file and upload to Twitter
# TODO: Delete temp file after published - or better yet, find way to update status from buffer.

def sendTweet(current_chyron, image):
    print('Tweeting: ' + current_chyron['text'])
    temp_path = './temp.jpg'
    cv2.imwrite(temp_path, image)
    print(temp_path)
    print(api.update_with_media(filename=temp_path, status=current_chyron['text']))

# Counter to keep track of commercial length
commercial_airing_for = 0

# Main loop - if file read is successful, might crash if stream stalls. Need to handle restarts.
while success:
    
    frameId = int(round(vidcap.get(1)))
    success, image = vidcap.read()

    # Setup dimensions based on current resolution using the first frame.
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
    
    # Process each second to keep track of commercials.
    if frameId % (fps * 1) == 0:
        commercialDetect = image[commercialStart:commercialEnd,commercialLeft:commercialRight]
        iscommercial = pytesseract.image_to_string(Image.fromarray(commercialDetect)).lower()

        if 'am' in iscommercial or 'pm' in iscommercial:
            commercial_airing_for = 0
        else: commercial_airing_for += 1
        # cv2.imshow('frame', image)

    # Read chyron every interval period.
    if frameId % multiplier == 0:
        
        # Extract chyron image and text using Tesseract
        lowerThird = image[lowerThirdStart:lowerThirdEnd,lowerThirdLeft:lowerThirdRight]
        text = pytesseract.image_to_string(Image.fromarray(lowerThird))

        # Cleanup non-ascii characters and correct for the letter O which gets incorrectly recognized as a 0
        # TODO: Train tesseract on the chyron font to improve detection.
        text = text.decode('utf-8')#.encode('ascii',errors='ignore')
        text = re.sub(r'(?![A-Z])?0(?![A-Z])?', 'O', text).replace("'IT", 'TT')
    
        if len(text) > 4:
            # Text has to be capitalized and show hasn't aired for 10 seconds (based on timestamp in the bottom right corner of the screen.)
            if text.isupper() and commercial_airing_for < 10:
                text = text.replace('\n', ' ') # multi line chyron

                # Log chyron
                print('CHYRON: ' + text)
                
                # Update DB & Tweet if required.
                # updateDB(text, image)
            # Halt if commercial has been detected for 10 consecutive seconds.
            elif commercial_airing_for > 10:
                print('Commercial has been airing for: ' + str(commercial_airing_for) + ' seconds.')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Video capture and kill any open openCV windows.
vidcap.release()
cv2.destroyAllWindows()