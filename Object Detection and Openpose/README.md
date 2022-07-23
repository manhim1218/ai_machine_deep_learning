# Tensorflow Object Detection / Openpose

This respository provides general computer vision algorithms for object detection, pose detection and human/object tracking.  

## Tensorflow Object Detection 
### General Description: 
The ssd_mobilenet_v2 oid v4 model is a pretrained model based on OpenImage version 4. It is capable of detecting 601 classes. Supporting objects are shown in objects.names.txt. This tensorflow object detection API, created by Google, is an opensource framework which is very useful to the computer vision project. 

The ssd_inception_v2_coco_2017 model is a pretrained model based on CoCo dataset 2017. This model is capable of detecting 80 classes. However, this model detect human, kitchen utensils or other small predefined small objects better than the ssd_mobilenet_v2 oid v4 model.

Combination of two models running simultaneously together will optimise the object detection capability. Threshold can be set differently to control false positive and false negative. 

## Tensorflow / Tensorflow Lite Openpose - detecting human joints and tracking human
### General Description: 
This project implements Movenet pose estimation model, which is the latest pose estimation model from tensorflow, to detect keypoints of a human based on the confidence score from 0 to 1. The score indicates the probability of detecting a keypoint in a certain position, such as hip or ankle. There are two versions of multipose algorithms available at this repository, one is tensorflow version (Object_detection_openpose.py) and the other one is tensorflow lite version (movenet folder). 

### Tensorflow Lite Openpose - human track
The tensorflow lite version is capable of tracking human and detecting human joints at the same time. 

# Object Tracker
### General Description: 
This project implements OpenCV object detector API including 'BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW' and 'CSRT'. Each detector has its own strength and weakness. My algorithm here is to use multiple detectors together (based on your preference) to take their strenths at one go. MobileNetSSD caffemodel is used here to detect object.  

### Demo of object detection and openpose
<p align="center"><img src="ironman.gif"\></p>
