from collections import OrderedDict
import pickle
from datetime import datetime
import requests
from flask import Flask, jsonify, request, redirect,render_template,url_for
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
from ast import literal_eval
import matplotlib.pyplot as plt
import urllib
from tqdm.notebook import tqdm
import glob
from PIL import Image,ImageOps
import os
import numpy as np 
import pandas as pd 
import tensorflow
import random
import cv2
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import warnings
from keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.models import load_model

import numpy as np
from flask import session
# Program to generate a random number between 0 and 9

# importing the random module
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def load_images_from_folder(folder):
	images = []
	dirs = os.listdir(folder)

	for filename in dirs:
		if os.path.isfile(folder+filename):
			im = Image.open(folder+filename)
			imResize = im.resize((224,224), Image.ANTIALIAS)
			imResize = np.array(imResize)
			if imResize is not None:
				images.append(imResize)
	return images


def prepareData(parentPath):
	
	# Make a list of all the 0 label and 1 label images for train, val, and test sets

	all0_path = list()
	

	all1_path = list()
	

	all0_path.append(parentPath+'/0/')
	all1_path.append(parentPath+'/1/')


	# Read images into respective lists
	

	allX = list()
	allY = list()
	xTrain = list()
	yTrain = list()
	xVal = list()
	yVal = list()
	xTest = list()
	yTest = list()

	print('Class 0, reading started..\n\n')
	

	tempImgs = list()
	tempImgs = load_images_from_folder(all0_path[0])
	for i in range(len(tempImgs)):
		allX.append(tempImgs[i])
		allY.append(0)

	# Prepare all data for 1 class
	print('Class 1, reading started..\n\n')
	tempImgs = list()
	tempImgs = load_images_from_folder(all1_path[0])
	for i in range(len(tempImgs)):
		allX.append(tempImgs[i])
		allY.append(1)

	xTrain,xTest,yTrain,yTest = train_test_split(allX,allY, test_size=0.20, random_state=23, shuffle = True)

	xpTrain,xVal,ypTrain,yVal = train_test_split(xTrain,yTrain, test_size=0.10, random_state=23, shuffle = True)

	return xpTrain, ypTrain, xVal, yVal, xTest, yTest
from keras.applications.inception_v3 import InceptionV3
def InceptionV3_transfer_actual(input_shape):
  res = InceptionV3(weights=None, include_top=False, input_shape=input_shape)
  for layers in res.layers:
    layers.trainable = True

  model = Sequential()
  model.add(res)
  
  model.add(Flatten())

  ## The following is the vanilla VGG16 architecture's FC layers
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=4096,activation="relu"))
  model.add(Dense(units=1000,activation="relu"))
  model.add(Dense(units=1, activation="sigmoid"))

  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  model.summary()
  return model
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
model1 = InceptionV3_transfer_actual(input_shape)
model1.load_weights('inception_v3.h5') 


labels=["Emphysema","Fibrosis","Ground Glass","nodules"]

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
import pickle 




from random import seed
from random import randint
#to generate seed number
seed(101)

@app.route('/')
def home():
	return render_template('./index.html')
@app.route('/index')
def index():
	return render_template('./index.html')

@app.route('/res')
def res():
	return render_template('./result.html')

@app.route('/fileupload', methods=["GET", "POST"])
def fileupload():
    
    imageurl = request.files['fileurl']
    imageurl.save("1.jpg")
    value = str(randint(0,90))+".jpg"
    img = cv2.imread("1.jpg")
    cv2.imwrite("static/images/"+str(value), img)
    
    imResize = cv2.resize(img, (224, 224))
    imResize = np.array(imResize)
    normalize = imResize / 255
    data=[]
    data.append(normalize)
    data.append(normalize)
    data.append(normalize)
    print(normalize.shape)
    testPreds1 = model1.predict(np.asarray(data))
    res="ulcer"
    prob="Healthskin("+str((1-testPreds1[0])*100)+"%)"+"Diabetic Food Ulcer("+str(testPreds1[0]*100)+"%)"
    if(testPreds1[0]<=0.5):
        res="Healthskin(Result)"
       
        
    else:
        res="Diabetic Food Ulcer(Result)"
       
    return render_template('./result.html',val=value,res=res,prob=prob)
  
 
        
if __name__ == '__main__':
       
       app.run(debug=False)
