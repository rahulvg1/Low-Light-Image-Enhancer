


import numpy as np
import cv2, glob
from skimage.transform import resize
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img, load_img

def L1(yhat, y):
    loss = np.sum(np.abs(yhat-y),axis = 0)
    return loss

model = load_model('Model/3.1.h5',custom_objects={'L1':L1})
model.load_weights('Model/Weights/3.1._wt.h5')

image = cv2.imread('image.png')
t= []


image = cv2.resize(image, (256, 256)).astype('float32')
image =img/255
t.append(image)
image = np.asarray(t)
pred = model.predict(image)

ip= np.squeeze(pred)*255*-1

cv2.imwrite("output.png",ip)
