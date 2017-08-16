import sys
import subprocess
from threading  import Thread
import threading
import time
import os

from Queue import Queue, Empty

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

last_alive = time.time()

def startParser():
    logging.info('Starting chyron detector.')
    process = subprocess.Popen('exec python ./parseVideo.py', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    return process

def checkOutput():
    global parserProcess, outputQueue
    out = parserProcess.stdout
    for line in iter(out.readline, b''):
        outputQueue.put(line.strip())
    out.close()

def monitorParser():
    
    while True:
        global last_alive
        global outputQueue
        try:  line = outputQueue.get_nowait() # or q.get(timeout=.1)
        except Empty:
            stalled_for = time.time() - last_alive
            #logging.info('No output for %f', stalled_for)

            if stalled_for > 40:
                #Process has stalled for longer than 10 seconds.
                logging.info('Parser stalled. Restarting')
                restart()
		time.sleep(10)
        else: # got line
            logging.info('Got line: %s', line)
            last_alive = time.time()
        #threading.Timer(1, monitorParser).start()

def restart():
    global parserProcess

        # Parser is still alive, probably stalled - kill it.
    try:
	parserProcess.kill()
    except OSError:
          # silently fail if the subprocess has exited already
        pass
    start(restart=True) # Restart the parser.

def start(restart=False):
    global parserProcess, outputQueue
    parserProcess = startParser()
    outputQueue = Queue()
    
    outputThread = Thread(target=checkOutput)
    outputThread.daemon = True # thread dies with the program
    outputThread.start()
    
    if not restart: monitorParser()
    logging.info('Parser dead. Restart.')

if __name__ == '__main__':
    start()
