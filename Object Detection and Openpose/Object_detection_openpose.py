#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

t0 = time.time()
config_file = './graph.pbtxt.txt'
frozen_model = './frozen_inference_graph.pb'

model_object_detection = cv2.dnn.readNetFromTensorflow(frozen_model, config_file)

model = tf.saved_model.load('./')
movenet = model.signatures['serving_default']
t1 = time.time()


classLabels = []
file_name = './objects.names.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels)


KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

key_list = list(KEYPOINT_DICT.keys())
val_list = list(KEYPOINT_DICT.values())


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            # print(kp)
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


def group_similar_objects(detection):
    classID = int(detection[1])
    if classID == 405 or classID == 554:  # hand bag (405), (Luggage and bag) 554
        classe = "Bag"
    if classID == 328: # Plastic bag for trash bag?
        classe = "Trash bag" 
    if classID == 312: # change plate to dish
        classe = "Dish"
    if classID == 196: # change flying disc to disk
        classe = "Disk"  
    if classID == 293 or classID == 237: # Bowl(293), Mixing bowl (237)
        classe = "Bowl" 
    if classID == 179 or classID == 401: #Coffee cup(179), Measuring cup(401)
        classe = "Cup"
    if classID == 191: # change paper towel to napkin
        classe = "Napkin"
    if classID == 325 or classID == 285: # Knife 285, Kitchen Knife 325
        classe = "Knife"
    if classID == 395: # Picnic Basket
        classe = "Basket"
    if classID == 444:
        classe = "Rubbish Bin"
    else:
        classe = classLabels[classID - 1]
    return classe

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
                
    # Resize image
    img = frame.copy()

    #192 x 256 depends on the frame width and height of your camera, we need to change the aspect ratio of width for matching keypoint in the camera. Height is fixed in 256 according to tensorflow. In this case my webcam camera is 480 x 640, that means 480/640 * 256 = 192, so change the width to 192 for matching the aspect ratio.
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160,256) 
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))

    loop_through_people(frame, keypoints_with_scores, EDGES, 0.3)
    imageBGR = np.array(frame)
    blob = cv2.dnn.blobFromImage(imageBGR, size=(300, 300), swapRB=False, crop=False)
    model_object_detection.setInput(blob)
    outputs = model_object_detection.forward()
    
    width = imageBGR.shape[1]
    height = imageBGR.shape[0]
        
    for detection in outputs[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.3:
            classID = int(detection[1])
            #print(classID)
            
            # chairs_taken_or_empty(outputs)
            
            classe = group_similar_objects(detection)
            #print(classe)

            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            
            cropped = frame[int(top):int(bottom), int(left):int(right)] 
            cv2.rectangle(frame, (left, top), (right, bottom), (255,0,0),2)
            cv2.putText(frame, str(classe), (int(left), int(top)), font, fontScale=font_scale, color=(0,255,0), thickness =3)
    
    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


