import cv2
import numpy as np
import matplotlib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import os
import h5py
import keras
import numpy as np

##################################################
input_size = (28,28)
##################################################
model =load_model('digimodel_ready.h5')
#print(model.summary())

def preprocess(img):
    # reshape into a single sample with 1 channel

    img = img.astype('float32')
    img = img / 255.0
    img = cv2.resize(img,input_size)
    img = img.reshape(1,28,28,1)
    # prepare pixel data
    return img



# path = 'dday.png'
# frame = cv2.imread(path,0)
# img = preprocess(frame)
# print(img.shape)
# pc = model.predict_classes(img, verbose=0)
# p = model.predict(img, verbose=0)
#
# print('class' + str(pc) + ' prob' + str(np.max(p)))
# cv2.putText(frame, 'class' + str(pc) + 'prob' + str(np.max(p)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255),
#                 1)
# # cv2.resize(gray, size=(400,300))
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    success , frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    # Our operations on the frame come here
    img = preprocess(thresh)
    # Display the resulting frame

    p= model.predict(img,verbose=0)
    pc= model.predict_classes(img,verbose=0)

    tot = np.max(p)
    if tot >= 0.8:
        print('class' + str(pc) + 'prob' + str(p))
        cv2.putText(frame,'class' + str(pc) + ' prob' + str(np.max(p)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 54, 255), 1)
    #cv2.resize(gray, size=(400,300))
    cv2.imshow('frame',frame)
    cv2.imshow('filter',thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()