FROM victorhcm/opencv

RUN apt-get update
RUN apt-get install -y libffi-dev libssl-dev
RUN pip install --upgrade pip
RUN pip install pyopenssl

RUN apt-get -y build-dep python-imaging
RUN apt-get install -y libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
RUN pip install Pillow

RUN apt-get install libfreetype6 libfreetype6-dev zlib1g-dev

RUN apt-get -y install tesseract-ocr

RUN pip install pytesseract Pillow streamlink tinydb tweepy