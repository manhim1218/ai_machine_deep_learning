# Tensorflow Object Detection / Openpose

This respository now only provides algorithms for general usage on computer. 
Examples of implementating on ROS (Pepper Robot) will be available after July 2022. 

## Tensorflow Object Detection 
### General Description: 
The ssd_mobilenet_v2 oid v4 model is capable of detecting 601 classes. Supporting objects are shown in objects.names.txt. This tensorflow object detection API, created by Google, is an opensource framework which is very useful to the computer vision project. 

## Tensorflow Openpose
### General Description: 
This project implements Movenet pose estimation model, which is the latest pose estimation model from tensorflow, to detect keypoints of a human based on the confidence score from 0 to 1. The score indicates the probability of detecting a keypoint in a certain position, such as hip or ankle. 

# Object Tracker
### General Description: 
This project implements OpenCV object detector API including 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW' and 'CSRT'. Each detector has its own strength and weakness. My algorithm here is to use multiple detectors together (based on your preference). MobileNetSSD caffemodel is used here to detect object.  

### Demo of object detection and openpose
<p align="center"><img src="ironman.gif"\></p>
