#!/usr/bin/env python3

import torch
import cv2
import os
import requests

# Constants
# FRAME_SOURCE = '/home/pi/fcs/image.png' #change home to pi
FRAME_SOURCE = './helmet.jpg'
API_URL = 'http://localhost:8080/'
SDD_IMAGES = "/home/pi/sdd/" #change home to pi

count = 0

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

img = None

# Inference

def send_notification():
    post_data = {
            "message":"violated"
            }
    requests.post(API_URL+"alert",json=post_data)

def send_revert():
    post_data = {
            "message":"n"
            }
    requests.post(API_URL+"alert",json=post_data)

def distance_of_two(cord_1,cord_2):
    global count
    if (abs(cord_1[0] - cord_2[0]) < 0.6):
        print("voilated")
        cv2.imwrite(SDD_IMAGES+str(count)+".png",img)
        count += 1
        send_notification()
    else:
        send_revert()

def take_photo():
    img = 

while True:
    img = cv2.imread(FRAME_SOURCE)
    print(type(img))
    try:
        results = model(img)
        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        print("coords ",cord_thres)
        if len(labels) == 0:
            send_revert()
        for i in range(0,len(labels)-1):
            if (labels[i] == 0):
                distance_of_two(cord_thres[i],cord_thres[i+1])
    except:
        print("none")