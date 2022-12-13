import cv2
import numpy as np
import face_recognition

# step 1
# loading the image
imgHarry = face_recognition.load_image_file('../imagesTrain/harrystyles.jpg')
# converting from BGR to RGB
imgHarry = cv2.cvtColor(imgHarry,cv2.COLOR_BGR2RGB)

# test image
imgTest = face_recognition.load_image_file('../imagesTrain/harrystylestest.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


# step 2
# finding the faces in image and their encodings
faceLoc = face_recognition.face_locations(imgHarry)[0] # [0] go zema samo prviot element
encodeHarry = face_recognition.face_encodings(imgHarry)[0]
# four coordinates of a square
cv2.rectangle(imgHarry,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # detection of face, color and thickness

# only the encodings for testing, not the location
faceLocTest = face_recognition.face_locations(imgTest)[0] # [0] only takes the first element
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


# step 3
# comparing the faces and finding the distance between them
# 128 measuremnts for each face
results = face_recognition.compare_faces([encodeHarry],encodeTest) # a list of faces
faceDis = face_recognition.face_distance([encodeHarry],encodeTest) # distance
print(results,faceDis) # print results and distance
# print results on the image, distance on 2 decimal places, origin, font, scale, color red, thickness
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

# displaying results
cv2.imshow('Harry Styles', imgHarry)
cv2.imshow('Harry Styles Test', imgTest)
cv2.waitKey(0)
