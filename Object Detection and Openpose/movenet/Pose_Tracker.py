import argparse
import logging
import sys
import time
import cv2
from ml import Movenet
# from ml import MoveNetMultiPose
# from ml import Posenet
import utils
import rospy
from std_msgs.msg import String
import numpy as np
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point32
from perception_utils.utils import *
from perception_pepper.msg import Object

sys.path.append("./utils")

DISPLAY = 1
CV_BRIDGE_USED = 1

W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple
CYAN = '\033[96m'

if CV_BRIDGE_USED == 1:
    print("Using original CV_BRIDGE")
    from cv_bridge import CvBridge, CvBridgeError
else:
    print("Using Util CV_BRIDGE")
    from nobridgeUtil_python3 import imgmsg_to_cv2, imgmsg_to_cv2_nocolor, cv2_to_imgmsg


class OpenPose_Tracker():
    def __init__(self):
        rospy.init_node('pose_tracker', anonymous=True)

        if CV_BRIDGE_USED == 1:
            # Create the cv_bridge object
            self.bridge = CvBridge()

        self.initCameras()

        self.initDisplay()

        self.mean_learn_feature = np.empty(0)

        self.mean_observed_feature = np.empty(0)

        self.model = 'movenet_lightning'
        self.tracker = 'bounding_box'
        self.classifier = False
        self.label_file = 'labels.txt'
        self.cameraId = 0


        self.first_time = True

        self.pub_cv = rospy.Publisher('/roboBreizh_detector/openpose_tracker', Image, queue_size=10)

        self.output_publisher = rospy.Publisher("/perception_pepper/tracking_object", Object, queue_size=10)
       
        # spin
        print("Waiting for image topics...")
        rate = rospy.Rate(5)
        rospy.spin()

    def initCameras(self):
        self.camera_info_sub = message_filters.Subscriber(
            '/naoqi_driver/camera/front/camera_info', CameraInfo)
        self.image_sub = message_filters.Subscriber(
            "/naoqi_driver/camera/front/image_raw", Image)
        self.depth_sub = message_filters.Subscriber(
            "/naoqi_driver/camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.camera_info_sub], queue_size=10, slop=0.5, allow_headerless=True)
        self.ts.registerCallback(self.image_callback,)


    def learn(self, estimation_model: str, tracker_type: str, width: int, height: int, frame, image, depth_image, camera_info) -> None:

        # Notify users that tracker is only enabled for MoveNet MultiPose model.
        if tracker_type and (estimation_model != 'movenet_multipose'):
            logging.warning(
                'No tracker will be used as tracker can only be enabled for '
                'MoveNet MultiPose model.')

        # Initialize the pose estimator selected.
        if estimation_model in ['movenet_lightning', 'movenet_thunder']:
            pose_detector = Movenet(estimation_model)
        elif estimation_model == 'posenet':
            pose_detector = Posenet(estimation_model)
        elif estimation_model == 'movenet_multipose':
            pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
        else:
            sys.exit('ERROR: Model is not supported.')

        # Variables to calculate FPS
        counter, fps = 0, 0
        start_time = time.time()

        # Visualization parameters
        row_size = 20  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 255)  # red
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        counter += 1
        image = cv2.flip(image, 1)

        if estimation_model == 'movenet_multipose':
            list_persons = pose_detector.detect(image)
        else:
            list_persons = [pose_detector.detect(image)]

        image_feature = pose_detector.getImageFeature()

        self.mean_learn_feature = np.mean(image_feature,axis=0)
        
        # Draw keypoints and edges on input image
        frame, start_x, start_y, end_x, end_y = utils.visualize_learn(image, list_persons)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = ' + str(int(fps))
        text_location = (left_margin, row_size)
        cv2.putText(frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        pose_message_learn = self.bridge.cv2_to_imgmsg(frame, "bgr8")

        dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistance(
            depth_image, start_x, end_y, start_y, end_x, camera_info)

        obj_learn = Object()
        obj_learn.label = 'Person'
        obj_learn.coord = Point32(point_x, point_y, point_z)
        obj_learn.distance = float(dist)
        print(obj_learn)
        self.pub_cv.publish(pose_message_learn)
        self.output_publisher.publish(obj_learn)

    def run(self, estimation_model: str, tracker_type: str, width: int, height: int, frame, image, depth_image, camera_info) -> None:

        # Notify users that tracker is only enabled for MoveNet MultiPose model.
        if tracker_type and (estimation_model != 'movenet_multipose'):
            logging.warning(
                'No tracker will be used as tracker can only be enabled for '
                'MoveNet MultiPose model.')

        # Initialize the pose estimator selected.
        if estimation_model == 'movenet_multipose':
            pose_detector = MoveNetMultiPose(estimation_model, tracker_type)
        else:
            sys.exit('ERROR: Model is not supported.')

        # Variables to calculate FPS
        counter, fps = 0, 0
        start_time = time.time()

        # Visualization parameters
        row_size = 20  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 255)  # red
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        counter += 1
        image = cv2.flip(image, 1)

        if estimation_model == 'movenet_multipose':
            # Run pose estimation using a MultiPose model.
            list_persons = pose_detector.detect(image)
        else:
            # Run pose estimation using a SinglePose model, and wrap the result in an
            # array.
            list_persons = [pose_detector.detect(image)]


        image_feature_observe = pose_detector.getImageFeature()

        self.mean_observed_feature = np.mean(image_feature_observe,axis=0)

        self.mean_learn_feature = self.mean_learn_feature.flatten()
        self.mean_observed_feature = self.mean_observed_feature.flatten()

        cosine = np.dot(self.mean_learn_feature,self.mean_observed_feature)/(np.linalg.norm(self.mean_learn_feature)*np.linalg.norm(self.mean_observed_feature))

        frame, start_x, start_y, end_x, end_y = utils.visualize_observe(cosine, image, list_persons)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = ' + str(int(fps))
        text_location = (left_margin, row_size)
        cv2.putText(frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)


        pose_message = self.bridge.cv2_to_imgmsg(frame, "bgr8")

        dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistance(
            depth_image, start_x, end_y, start_y, end_x, camera_info)

        obj = Object()
        obj.label = 'Person'
        obj.coord = Point32(point_x, point_y, point_z)
        obj.distance = float(dist)
        print(obj)
        self.pub_cv.publish(pose_message)
        self.output_publisher.publish(obj)


    def image_callback(self, ros_image, ros_depth_image, camera_info):

        # Use cv_bridge() to convert the ROS image to OpenCV format
        img = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        depth_image = self.bridge.imgmsg_to_cv2(ros_depth_image, "32FC1")

        depth_array = np.array(depth_image, dtype=np.float32)
        cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
        depth_8 = (depth_array * 255).round().astype(np.uint8)
        cv_depth = np.zeros_like(img)
        cv_depth[:, :, 0] = depth_8
        cv_depth[:, :, 1] = depth_8
        cv_depth[:, :, 2] = depth_8

        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        frame = np.array(img, dtype=np.uint8)
        print(frame.shape)
        frame = cv2.resize(frame, (640, 480))

        frameWidth = img.shape[1]
        frameHeight = img.shape[0]

        if self.first_time:
            self.learn(self.model, self.tracker, frameWidth, frameHeight, frame, img, depth_image, camera_info)
            self.first_time = False
        if not self.first_time:
            self.run(self.model, self.tracker, frameWidth, frameHeight, frame, img, depth_image, camera_info)

if __name__ == '__main__':
    OpenPose_Tracker()
