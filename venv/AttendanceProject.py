import cv2
import numpy as np
import face_recognition
import os
from datetime import *
# from TrainNetwork import *
from General import *


# d = get_date()
# path = 'Attendance ' + d + '.txt'
path = 'Attendance.txt'
all_face_encodings = load_encodings('dataset_faces.dat')


# mark the attendance with name
def markAttendance(name):
    with open(path, 'r+') as f:  # r+ is to read and write at the same time
        myDataList = f.readlines()  # list
        # print(myDataList)
        nameList = []
        for line in myDataList:  # for every entry/line
            entry = line.split(',')  # split them based on comma
            nameList.append(entry[0])  # name is our first element [0], all the names in the list
        # is the current name present or not
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')  # time format
            f.writelines(f'\n{name}, {dtString}')


# print the number of encodings
# encodeListKnown = findEncodings(images)
# print(len(encodeListKnown)) #print('Encoding Complete')

# step 4
# FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
    # cap = np.array(ImageGrab.grab(bbox))
    # cap = cv2.cvtColor(cap, cv2.COLOR_RGB2BGR)
    # return cap


# find the match with webcam
cap = cv2.VideoCapture(0)


while True:  # to get each frame 1 by 1
    # add Name, Time at the top of the file
    with open(path, 'r+') as f:  # r+ is to read and write at the same time
        f.write('Name, Time\n')
    # img = captureScreen()
    success, img = cap.read()
    # reduce the size bc its in realtime, image/no pixels size/scale(1/4)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # convert to RGB

    # find the locations
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # imgS=small image

    # encodeListK = encodings_to_set('Encodings.txt')
    # iterate through all the faces in the current frame and compare them with
    # all the encodings we found before
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):  # in one loop
        matches = face_recognition.compare_faces(all_face_encodings, encodeFace)  # comparing with known
        faceDis = face_recognition.face_distance(all_face_encodings, encodeFace)  # lowest distance is best match
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        # display bounding box and write their name
        if faceDis[matchIndex] < 0.5:  # if the result is smaller than 0.5 -> mark it as present
            person_name = classNames[matchIndex].upper()
            markAttendance(person_name)
        else:
            person_name = 'Unknown'
            # print(name)
            # face location
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # bc we have scaled it before according to the small img
        # draw a rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
        # write name
        conf = face_distance_to_conf(faceDis[0])*100
        cv2.putText(img, f'{person_name} {round(conf,2)}%', (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        # when a match is found, call the function to mark the attendance
        # markAttendance(name)

    cv2.imshow('Iriun Webcam', img)
    cv2.waitKey(1)

    text_to_sheet(path)
    #delete_file_contents(path)