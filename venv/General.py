import cv2
import numpy as np
import face_recognition
import os
from datetime import *
import pandas as pd
import re
import pickle
from TrainNetwork import *
import math


def get_date():
    today = date.today()
    d = today.strftime("%d %B, %Y")
    return d


# convert the text file to Excel file
def text_to_sheet(path_name):
    d2 = get_date()
    df = pd.read_csv(path_name)  # can replace with df = pd.read_table('input.txt') for '\t'
    df.to_excel('Attendance ' + d2 + '.xlsx', sheet_name=d2)
    df.to_csv('Attendance ' + d2 + '.csv')


# Delete the contents of the file
def delete_file_contents(path):
    open(path, 'w').close()


def load_encodings(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def face_distance_to_conf(face_distance, face_match_threshold=0.5):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


"""for el in encodeListKnown:
    print(el)"""
# append_encodings(encodeListKnown)
# print(encodings_to_set('Encodings.txt'))
