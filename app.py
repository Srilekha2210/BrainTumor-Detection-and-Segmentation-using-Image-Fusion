from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, render_template, request, redirect, Response


from keras.preprocessing.image import load_img, img_to_array

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

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import os
from datetime import datetime
from flask import Flask, render_template, request, url_for
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import cv2
import imageio
import scipy.ndimage as ndi
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg19
import pywt
import pywt.data
#from skimage.morphology import extrema
#from skimage.morphology import watershed as skwater



app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Our methods
def convertToIntList(arr):
    result=[]
    for q in arr.strip(']][[').split('],['):
        x=[]
        for i in q.split(','):
            x.append(int(i,10))
        result.append(x)
    return result

def procrustes(X, Y, scaling=True, reflection='best'):
    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #rot =1
    #scale=2
    #translate=3
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

# VGG19 CNN For Fusion
class VGG19(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(VGG19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        if device == "cuda":
            self.features = nn.ModuleList(features).cuda().eval()
        else:
            self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        feature_maps = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == 3:
                feature_maps.append(x)
        return feature_maps

class Fusion:
    def __init__(self, input):
        """
        Class Fusion constructor

        Instance Variables:
            self.images: input images
            self.model: CNN model, default=vgg19
            self.device: either 'cuda' or 'cpu'
        """
        self.input_images = input
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGG19(self.device)

    def fuse(self):
        """
        A top level method which fuse self.images
        """
        # Convert all images to YCbCr format
        self.normalized_images = [-1 for img in self.input_images]
        self.YCbCr_images = [-1 for img in self.input_images]
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx] = self._RGB_to_YCbCr(img)
                self.normalized_images[idx] = self.YCbCr_images[idx][:, :, 0]
            else:
                self.normalized_images[idx] = img / 255.
        # Transfer all images to PyTorch tensors
        self._tranfer_to_tensor()
        # Perform fuse strategy
        fused_img = self._fuse()[:, :, 0]
        # Reconstruct fused image given rgb input images
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx][:, :, 0] = fused_img
                fused_img = self._YCbCr_to_RGB(self.YCbCr_images[idx])
                fused_img = np.clip(fused_img, 0, 1)

        return (fused_img * 255).astype(np.uint8)
        # return fused_img

    def _fuse(self):
        """
        Perform fusion algorithm
        """
        with torch.no_grad():

            imgs_sum_maps = [-1 for tensor_img in self.images_to_tensors]
            for idx, tensor_img in enumerate(self.images_to_tensors):
                imgs_sum_maps[idx] = []
                feature_maps = self.model(tensor_img)
                for feature_map in feature_maps:
                    sum_map = torch.sum(feature_map, dim=1, keepdim=True)
                    imgs_sum_maps[idx].append(sum_map)

            max_fusion = None
            for sum_maps in zip(*imgs_sum_maps):
                features = torch.cat(sum_maps, dim=1)
                weights = self._softmax(F.interpolate(features,
                                        size=self.images_to_tensors[0].shape[2:]))
                weights = F.interpolate(weights,
                                        size=self.images_to_tensors[0].shape[2:])
                current_fusion = torch.zeros(self.images_to_tensors[0].shape)
                for idx, tensor_img in enumerate(self.images_to_tensors):
                    current_fusion += tensor_img * weights[:,idx]
                if max_fusion is None:
                    max_fusion = current_fusion
                else:
                    max_fusion = torch.max(max_fusion, current_fusion)

            output = np.squeeze(max_fusion.cpu().numpy())
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            return output
        
        
    def _RGB_to_YCbCr(self, img_RGB):
            """
            A private method which converts an RGB image to YCrCb format
            """
            img_RGB = img_RGB.astype(np.float32) / 255.
            return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    def _YCbCr_to_RGB(self, img_YCbCr):
            """
            A private method which converts a YCrCb image to RGB format
            """
            img_YCbCr = img_YCbCr.astype(np.float32)
            return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

    def _is_gray(self, img):
            """
            A private method which returns True if image is gray, otherwise False
            """
            if len(img.shape) < 3:
                return True
            if img.shape[2] == 1:
                return True
            b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
            if (b == g).all() and (b == r).all():
                return True
            return False

    def _softmax(self, tensor):
            """
            A private method which compute softmax ouput of a given tensor
            """
            tensor = torch.exp(tensor)
            tensor = tensor / tensor.sum(dim=1, keepdim=True)
            return tensor

    def _tranfer_to_tensor(self):
            """
            A private method to transfer all input images to PyTorch tensors
            """
            self.images_to_tensors = []
            for image in self.normalized_images:
                np_input = image.astype(np.float32)
                if np_input.ndim == 2:
                    np_input = np.repeat(np_input[None, None], 3, axis=1)
                else:
                    np_input = np.transpose(np_input, (2, 0, 1))[None]
                if self.device == "cuda":
                    self.images_to_tensors.append(torch.from_numpy(np_input).cuda())
                else:
                    self.images_to_tensors.append(torch.from_numpy(np_input)) 



UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path1 = 'model.h5' # load .h5 Model


#classes2 = {0:"APHIDS",1:"ARMYWORM",2:"BEETLE",3:"BOLLWORM",4:"GRASSHOPPER",5:"MITES",6:"MOSQUITO",7:"SAWFLY",8:"STEM BORER"}
model1 = load_model(model_path1)

model = torch.hub.load("ultralytics/yolov5", "custom", path = "best.pt", force_reload=True)

model.eval()
model.conf = 0.5  
model.iou = 0.45  

from io import BytesIO

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/index1')
def index1():
	return render_template('index1.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

    if request.method == "POST":

        print("Entered")
        target = os.path.join(APP_ROOT, 'static/')
        
        print("Entered here")
        mri_file=request.files['mri']
        ct_file=request.files['ct']

        destination1 = "/".join([target, "mri.jpg"])
        mri_file.save(destination1)

        destination2 = "/".join([target, "ct.jpg"])
        ct_file.save(destination2)

        ct = cv2.imread('static/ct.jpg', 0)
        mri = cv2.imread('static/mri.jpg', 0)

        input_images = []

        input_images.append(mri)
        input_images.append(ct)

        FU = Fusion(input_images)
        fusion_img = FU.fuse()
        # Write fusion image
        cv2.imwrite('static/img.jpg', fusion_img)
            
        
        file_path = 'static/img.jpg'
        print("@@ Predicting class......")

        image = load_img(file_path,target_size=(128,128))
        image = img_to_array(image)
        image = image/255
        image = np.expand_dims(image,axis=0)
        
        result = np.argmax(model1.predict(image))

        if result == 0:
            pred = 'NORMAL - NO BRAIN TUMOR!'
            return render_template('result.html', pred_output = pred, img_src='static/img.jpg')

        else:
            pred = 'ABNORMAL - BRAIN TUMOR!'
            return render_template('result1.html', pred_output = pred, img_src='static/img.jpg')
        
    return render_template('index.html')
        


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
        # results.render()  
        for img in results.render():
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")
    return render_template("index1.html")


    

   
if __name__ == '__main__':
    app.run(debug=False)