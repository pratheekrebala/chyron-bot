import cv2
import math
import numpy as np

#Load Video

videoFile = "sample.mp4"
vidcap = cv2.VideoCapture(videoFile)
success,image = vidcap.read()

# Get Video every five seconds (assume 30 fps)
seconds = 5
fps = 30 #vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

#Extract and Mask Chyron every 5 seconds.

dimensions = None

while success:
    frameId = int(round(vidcap.get(1))) #current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = vidcap.read()
    if frameId == 1:
        dimensions = dict(zip(('height', 'width', 'channels'), image.shape)) 
        lowerThirdStart = dimensions['height'] - int(dimensions['height'] * 0.25)
        lowerThirdEnd = dimensions['height'] - int(dimensions['height'] * 0.1)
        lowerThirdRight = dimensions['width'] - int(dimensions['width'] * 0.14)
    if frameId % multiplier == 0:
        lowerThird = image[lowerThirdStart:lowerThirdEnd,0:lowerThirdRight]
        gray = cv2.cvtColor(lowerThird,cv2.COLOR_BGR2RGB)
        lower_white = np.array([200,200,200], dtype=np.uint8)
        upper_white = np.array([255,255,255], dtype=np.uint8)
        mask = cv2.inRange(lowerThird, lower_white, upper_white)
        res = cv2.bitwise_and(lowerThird,lowerThird,mask = mask)
        cv2.imshow('frame', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()