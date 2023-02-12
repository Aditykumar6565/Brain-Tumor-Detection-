import cv2
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

##from flask import *
from werkzeug.utils import secure_filename
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
##Define the flask app
##app = Flask(__name__)

##Load trained model
model = load_model('braintumor-classification.h5')


img = cv2.imread('D:/BRAIN DATSET 31-0/BrainTUMOR-Classify/Dataset1-multi/Testing/meningioma_tumor/m(2).jpg')
img = cv2.resize(img,(150,150))
img_array = np.array(img)
img_array.shape

img_array = img_array.reshape(1,150,150,3)
img_array.shape

from tensorflow.keras.preprocessing import image
img = image.load_img('D:/BRAIN DATSET 31-0/BrainTUMOR-Classify/Dataset1-multi/Testing/meningioma_tumor/m(2).jpg')
plt.imshow(img,interpolation='nearest')
plt.show()

a=model.predict(img_array)
indices = a.argmax() + 1
print(indices)

if indices == 0:
 print("You have Glioma Type of Brain Tumor");
elif indices == 1:
    print("You have Meningioma Type of Brain Tumor");
elif indices == 2:
    print("You have No Brain Tumor");
elif indices == 3:
    print("You have Pituitary Type of Brain Tumor");

