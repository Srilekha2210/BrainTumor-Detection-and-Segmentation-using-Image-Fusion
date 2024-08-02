
import sys
import os
import glob
import re
import numpy as np

from PIL import Image
from pylab import *
from PIL import Image, ImageChops, ImageEnhance



# Keras
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt


np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from flask import Flask, render_template, request, redirect, Response

import pandas as pd
import numpy as np

import sys
import os
import glob
import re
import numpy as np


# Flask utils
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
from torchvision.models import detection
import sqlite3
import torch
from torchvision import models
from flask import Flask, render_template, request, redirect, Response

import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

from werkzeug.utils import secure_filename

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','tif'])

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = 'static/tempresaved.jpg'
    ELA_filename = 'static/tempela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality = quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    #ela_im.save(ELA_filename, 'JPEG', quality = quality)
    
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    #print(type(ela_im))
    return ela_im

model1 = load_model('model.h5')

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    
    print("Entered here")
    file = request.files['file'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("@@ Predicting class......")
    X_f = []
    X_f.append(np.array(convert_to_ela_image(file_path,90).resize((128, 128))).flatten() / 255.0)

    img = convert_to_ela_image(file_path,90)
    img.save('static/tempela.png', 'JPEG')


    X_f = np.array(X_f)
    X_f = X_f.reshape(-1, 128, 128, 3)
    y_pred_test = model1.predict(X_f)
    y_pred_test = np.argmax(y_pred_test,axis = 1)
    
    if y_pred_test[0] == 0:
        pred = 'The Given Input Image is Original!'

    else:
        pred = 'The Given Input Image is Tampered!'
    
    
    
              
    return render_template('result.html', pred_output = pred, img_src=UPLOAD_FOLDER + file.filename)

@app.route('/index1')
def index1():
	return render_template('index1.html')

model = torch.hub.load("ultralytics/yolov5", "custom", path = "best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45  

from io import BytesIO

def gen():
    """
    The function takes in a video stream from the webcam, runs it through the model, and returns the
    output of the model as a video stream
    """
    cap=cv2.VideoCapture(0)
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=415)
            results.print()  
            img = np.squeeze(results.render()) 
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        else:
            break
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    """
    It returns a response object that contains a generator function that yields a sequence of images
    :return: A response object with the gen() function as the body.
    """
    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route("/predict1", methods=["GET", "POST"])
def predict1():
    """
    The function takes in an image, runs it through the model, and then saves the output image to a
    static folder
    :return: The image is being returned.
    """
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=415)
        results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")
    return render_template("index1.html")

   
if __name__ == '__main__':
    app.run(debug=False)