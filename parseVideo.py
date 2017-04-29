import cv2
import math
import numpy as np
import tesseract
from PIL import Image
import pytesseract
import livestreamer
import string

live_cnn = 'https://www.youtube.com/watch?v=bxGy6ud54BE'
live_cnn_2 = 'https://www.youtube.com/watch?v=oamvsYd4AEM'
kellyanne_interview = 'https://www.youtube.com/watch?v=-otxWE6dBxk'
#streams = livestreamer.streams(live_cnn_2)

#Load Video

videoFile = "test.mp4"
vidcap = cv2.VideoCapture("test.mp4")#streams['360p'].url)
success,image = vidcap.read()

# Get Video every five seconds (assume 30 fps)
seconds = 5
fps = 30 #vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

allchars = string.maketrans('', '')
nonletter = allchars.translate(allchars, string.letters)

#Extract and Mask Chyron every 5 seconds.

dimensions = None

while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = vidcap.read()
    cv2.imshow('frame', image)
    if frameId == 1:
        dimensions = dict(zip(('height', 'width', 'channels'), image.shape)) 
        lowerThirdStart = dimensions['height'] - int(dimensions['height'] * 0.21)
        lowerThirdEnd = dimensions['height'] - int(dimensions['height'] * 0.09)
        lowerThirdRight = dimensions['width'] - int(dimensions['width'] * 0.15)
        lowerThirdLeft = int(dimensions['width'] * 0.05)
        #lowerThirdByLine = dimensions['width'] - int(dimensions['width'] * 0.13)
    if frameId % multiplier == 0:
        lowerThird = image[lowerThirdStart:lowerThirdEnd,lowerThirdLeft:lowerThirdRight]
        #lowerThirdInclusive = image[lowerThirdStart:lowerThirdByLine,lowerThirdLeft:lowerThirdRight]
        #gray = cv2.cvtColor(lowerThird,cv2.COLOR_BGR2RGB)
        #lower_white = np.array([200,200,200], dtype=np.uint8)
        #upper_white = np.array([255,255,255], dtype=np.uint8)
        #mask = cv2.inRange(lowerThird, lower_white, upper_white)
        #res = cv2.bitwise_and(lowerThird,lowerThird,mask = mask)
        text = pytesseract.image_to_string(Image.fromarray(lowerThird)).strip(nonletter)
        lines = text.splitlines()
        if len(text) > 4:
            if '\n' not in text and text.isupper():
                print('CHYRON:' + text)
            elif len(lines) > 1:
                pass
                #print('BY-LINE:' + lines[1])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()