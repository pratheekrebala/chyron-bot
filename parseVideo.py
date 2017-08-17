import os
import re
import json
import time
import string
from datetime import datetime, timedelta
import numpy
import signal
import sys


from imutils.video import WebcamVideoStream, VideoStream
from imutils.video import FPS
import argparse
import imutils

import logging

import cv2
from PIL import Image, ImageFilter

import streamlink
import pytesseract
from tinydb import TinyDB, Query
import tweepy

isproduction = False
liveTest = True
drawBoundaries = True
writeImages = True
network = "FNC"

logging.basicConfig(level=logging.NOTSET, format="%(asctime)-15s %(levelname)-8s %(message)s")

logger = logging.getLogger('')

#logger.addHandler(logging.StreamHandler(sys.stdout))
logger.addHandler(logging.FileHandler('./networks/' + network + '/output.log', mode='a+'))

logger = logging.getLogger(__name__)

def exit_gracefully(signum, frame):
    # Release Video capture and kill any open openCV windows.
    fps_m.stop()
    logging.info("Elasped time: {:.2f}".format(fps_m.elapsed()))
    logging.info("Approx. FPS: {:.2f}".format(fps_m.fps()))
    
    logging.info("Stopping stream and OpenCV")
    fvs.stop()
    cv2.destroyAllWindows()

    sys.exit(1)

    signal.signal(signal.SIGINT, exit_gracefully)

original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)

# Keeps track of airings and chyrons in a json file.
db = TinyDB('./chyron_db.json')

# Lookup Stream
with open('streams.json') as fp:
    stream_list = json.load(fp)
    #streams = streamlink.streams('https://www.youtube.com/watch?v=Q4sCA3QFqT8')
    logging.info('Live stream loaded from %s', 'streams.json.')

# Twitter Credentials
with open('credentials.json') as fp:    
    credentials = json.load(fp)
    logging.info('Twitter credentials loaded from %s', 'credentials.json.')

# Setup Tweepy
auth = tweepy.OAuthHandler(credentials['consumer_key'], credentials['consumer_secret'])
auth.set_access_token(credentials['access_key'], credentials['access_secret'])

api = tweepy.API(auth)

# videoFile = "test.mp4"
# Load Stream and pass onto openCV

logging.info("[INFO] starting video file thread...")
fvs = VideoStream(stream_list[network]).start()
time.sleep(1.0)
 
# start the FPS timer
fps_m = FPS().start()

#success,image = vidcap.read()

# Get Video every five seconds (assuming 30 fps)
seconds = 3
commercialSeconds = 2
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
    logging.info('Tweeting: %s', current_chyron['text'])
    temp_path = './temp.jpg'
    cv2.imwrite(temp_path, image)
    print(api.update_with_media(filename=temp_path, status=current_chyron['text']))

def getDimensions(image):
    #Take dimensions, read dimensions file and output three arrays
    # (incl image.)

    dimensions = dict(zip(('height', 'width', 'channels'), image.shape))

    with open('dimensions.json', 'r') as fp:
        locations = json.load(fp)[network]
        chyron = {
            "start": dimensions['height'] - int(dimensions['height'] * locations['chyron']['dimensions']['x1']),
            "end": dimensions['height'] - int(dimensions['height'] * locations['chyron']['dimensions']['x2']),
            "right": dimensions['width'] - int(dimensions['width'] * locations['chyron']['dimensions']['y1']),
            "left": dimensions['width'] - int(dimensions['width'] * locations['chyron']['dimensions']['y2']),
        }

        commercial = {
            "start": dimensions['height'] - int(dimensions['height'] * locations['commercial']['dimensions']['x1']),
            "end": dimensions['height'] - int(dimensions['height'] * locations['commercial']['dimensions']['x2']),
            "right": dimensions['width'] - int(dimensions['width'] * locations['commercial']['dimensions']['y1']),
            "left": dimensions['width'] - int(dimensions['width'] * locations['commercial']['dimensions']['y2']),
        }

        imageDim = {
            "start": dimensions['height'] - int(dimensions['height'] * locations['image']['dimensions']['x1']),
            "end": dimensions['height'] - int(dimensions['height'] * locations['image']['dimensions']['x2']),
            "right": dimensions['width'] - int(dimensions['width'] * locations['image']['dimensions']['y1']),
            "left": dimensions['width'] - int(dimensions['width'] * locations['image']['dimensions']['y2']),
        }

        return (chyron, commercial, imageDim)


# Counter to keep track of commercial length
commercial_airing_for = 0
commercialDetect =  None

def sharpenImage(img):
    #img = cv2.medianBlur(img,5)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #edgeMap = imutils.auto_canny(gray)

    #cv2.imshow("Original", logo)
    #cv2.imshow("Automatic Edge Map", edgeMap)

    kernel_sharpen_3 = numpy.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 6.0
    sharp = cv2.filter2D(gray, -1, kernel_sharpen_3)

    ret,thresh3 = cv2.threshold(sharp,220,255,cv2.THRESH_BINARY)

    (_, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.drawContours(img,cnts,-1,(255,255,255),3)
    #(h,s,v) = cv2.split(sharp)

    return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    # define range of white color in HSV
    # change it according to your need !
    lower_white = numpy.array([0,0,230], dtype=numpy.uint8)
    upper_white = numpy.array([180,255,255], dtype=numpy.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Bitwise-AND mask and original image
    img = cv2.bitwise_and(img, img, mask=mask)
    #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    return v
    
    img = numpy.concatenate((img, sharp, v), axis=0)
    return img

    return img


def extractBlack(img):

    # Try to sharpen image, not using for now.

    kernel_sharpen_3 = numpy.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

    #img = cv2.filter2D(img, -1, kernel_sharpen_3)
    #(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_black = numpy.array([0,0,0], dtype=numpy.uint8)
    upper_black = numpy.array([180,255,30], dtype=numpy.uint8)
    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)
    # Bitwise-AND mask and original image
    # img = cv2.bitwise_not(img,img, mask=mask)


    return mask
    
# Main loop - if file read is successful, might crash if stream stalls. Need to handle restarts.
frameId = 0
while not fvs.stopped:
    
    frameId += 1
    image = fvs.read()
    
    image = cv2.resize(image, (720, 480))
        
    if liveTest: chyronDim, commercialDim, imageDim = getDimensions(image)

    if frameId == 1:
        chyronDim, commercialDim, imageDim = getDimensions(image)
        logging.info('First frame received. Chyron location identified.')

    image = image[imageDim['start']:imageDim['end'],imageDim['right']:imageDim['left']]
    # Setup dimensions based on current resolution using the first frame.
    # Process each second to keep track of commercials.
    if frameId % (fps * commercialSeconds) == 0:
        commercialDetect = image[commercialDim['start']:commercialDim['end'],commercialDim['left']:commercialDim['right']]
        commercialDetect = sharpenImage(commercialDetect)
        iscommercial = pytesseract.image_to_string(Image.fromarray(commercialDetect)).lower()
        
        if 'am' in iscommercial or 'pm' in iscommercial or 'chan' in iscommercial or 'msn' in iscommercial or 'can' in iscommercial or 'caw' in iscommercial:
            commercial_airing_for = 0
        else: commercial_airing_for += 1

    # Read chyron every interval period.
    if frameId % multiplier == 0:
        lowerThird = image[chyronDim['start']:chyronDim['end'],chyronDim['right']:chyronDim['left']]
        if stream_list['sharpen'][network]: lowerThird = sharpenImage(lowerThird)
        # Extract chyron image and text using Tesseract
        text = pytesseract.image_to_string(Image.fromarray(lowerThird))

        if drawBoundaries:
            cv2.rectangle(image, (chyronDim['left'], chyronDim['start']), (chyronDim['right'], chyronDim['end']), (49,163,84), 3)
            cv2.rectangle(image, (commercialDim['left'], commercialDim['start']), (commercialDim['right'], commercialDim['end']), (255,255,0), 2)
        

        # Cleanup non-ascii characters and correct for the letter O which gets incorrectly recognized as a 0
        # TODO: Train tesseract on the chyron font to improve detection.
        text = text.encode('ascii',errors='ignore')
        text = re.sub(r'(?![A-Z])?0(?![A-Z])?', 'O', text).replace("'IT", 'TT')
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'_', '', text)
        text = re.sub('^[^a-zA-z]*|[^a-zA-Z]*$','', text)
        text = text.strip()
        chyron = None
        if len(text) > 4:
            # Text has to be capitalized and show hasn't aired for 10 seconds (based on timestamp in the bottom right corner of the screen.)
            lines = text.split('\n')
            if not text.isupper() and lines[0].isupper(): text = lines[0] # This handles the new condensed chyrons that span multiple lines.
            if text.isupper() and commercial_airing_for < 40:
                text = text.replace('\n', ' ') # multi line chyron
                chyron = text
                # Log chyron
                logging.info('CHYRON: %s', text)
                # Update DB & Tweet if required.
                if isproduction: updateDB(text, image)
            # Halt if commercial has been detected for 40 consecutive seconds.
            elif commercial_airing_for > 40:
                logging.info('Commercial has been airing for: %d seconds.', commercial_airing_for)
            else: logging.info('No Chyron Detected')
        else: logging.info('No Chyron Detected')

        if writeImages:
            #lowerThird = imutils.skeletonize(cv2.cvtColor(lowerThird, cv2.COLOR_BGR2GRAY), size=(5,5))
            cv2.imwrite('./networks/' + network + '/chyron.jpeg', lowerThird)
            cv2.imwrite('./networks/' + network + '/frame.jpeg', image)
            cv2.imwrite('./networks/' + network + '/commercial.jpeg', commercialDetect)
            chyron_meta = {'chyron': chyron, 'commercial': commercial_airing_for, 'chyron_src': text, 'commercial_src': iscommercial}
            with open('./networks/' + network + '/meta.json', 'w') as outfile:
                json.dump(chyron_meta, outfile)

exit_gracefully()