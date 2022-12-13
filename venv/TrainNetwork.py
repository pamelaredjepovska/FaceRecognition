import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from General import *

# step 1
# load images
# create a list that gets images and gets encoding automatically from folder so os biblioteka
path = '../imagesTrain2'
images = []  # list of all the images we import
classNames = []  # the names of all the images
# grab the images from the folder
myList = os.listdir(path)
# print(myList)


# step 2
# import each of the classes/names
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')  # cls is the image's name
    images.append(curImg)
    cls = cls.split(' ')[0] + ' ' + cls.split(' ')[1]
    # print(cls)
    classNames.append(os.path.splitext(cls)[0])  # append without the .jpg, just the first [0] element
# print(classNames)


# step 3, encoding process for each of the image with simple function
def findEncodings(images_list):
    encodeList = []  # empty list for all the encodings
    for img in images_list:  # loop through all the images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        encode = face_recognition.face_encodings(img)[0]  # find encodings
        encodeList.append(encode)  # append the encoding to the list
    return encodeList


def dump_encodings(encodings):
    with open('dataset_faces.dat', 'wb') as f:
        pickle.dump(encodings, f)


encodeListKnown = findEncodings(images)
dump_encodings(encodeListKnown)
