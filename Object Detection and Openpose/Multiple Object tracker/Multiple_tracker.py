import cv2
import sys
font_scale =3
font = cv2.FONT_HERSHEY_PLAIN
import numpy as np

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT']

tracker_type_3 = tracker_types[2]
tracker_type_4 = tracker_types[3]
tracker_type_6 = tracker_types[5]

tracker_3 = cv2.TrackerKCF_create()
tracker_4 = cv2.legacy.TrackerTLD_create()
tracker_6 = cv2.TrackerCSRT_create()

RESIZED_DIMENSIONS = (300, 300) # Dimensions that SSD was trained on. 
IMG_NORM_RATIO = 0.007843 # In grayscale a pixel can range between 0 and 255
 
# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 
        'MobileNetSSD_deploy.caffemodel')
 
# List of categories and classes
categories = { 0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 
               4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 
               9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 
              13: 'horse', 14: 'motorbike', 15: 'person', 
              16: 'pottedplant', 17: 'sheep', 18: 'sofa', 
              19: 'train', 20: 'tvmonitor'}
 
classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
            "bus", "car", "cat", "chair", "cow", 
           "diningtable",  "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
                      
# Create the bounding boxes
bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))
     
video = cv2.VideoCapture(0)

boolean = False
while (boolean == False):
    ok, frame = video.read()
    image_coding = frame.copy()
    
    width = frame.shape[1]
    height = frame.shape[0]
    print(width)
    print(height)
    
    frame_blob = cv2.dnn.blobFromImage(cv2.resize(image_coding, RESIZED_DIMENSIONS), 
                     IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
    
    neural_network.setInput(frame_blob)

    neural_network_output = neural_network.forward()
    for i in np.arange(0, neural_network_output.shape[2]):
        confidence = neural_network_output[0, 0, i, 2]
        if confidence > 0.30:
            idx = int(neural_network_output[0, 0, i, 1])
            detected_category = classes[idx]
            print(detected_category)
            try:
                if detected_category == 'person':
                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([width*0.5, height*0.6, width*0.6, height*0.6])
                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    bounding_box = (startX, startY, endX, endY)
                    boolean = True
                    break
            except IndexError:
                print("looking for Person")
                boolean = False

                
print(bounding_box)
bbox_3 = (bounding_box[0], bounding_box[1],
        bounding_box[2], bounding_box[3])

bbox_4 = (bounding_box[0], bounding_box[1],
        bounding_box[2], bounding_box[3])
print(type(bbox_4))
# bbox_5 = (bbox_person[0], bbox_person[1],
#         bbox_person[2], bbox_person[3])

bbox_6 = (bounding_box[0], bounding_box[1],
        bounding_box[2], bounding_box[3])

# Initialize tracker with first frame and bounding box
# ok_1 = tracker_1.init(frame, bbox_1)
# ok_2 = tracker_2.init(frame, bbox_2)
ok_3 = tracker_3.init(frame, bbox_3)
ok_4 = tracker_4.init(image_coding, bbox_4)
# ok_5 = tracker_5.init(frame, bbox_5)
ok_6 = tracker_6.init(image_coding, bbox_6)

while True:
    # Read a new frame
    ok, frame = video.read()

    image_coding = frame.copy()
    
    width = frame.shape[1]
    height = frame.shape[0]
    print(width)
    print(height)
    
    # Start timer
    timer = cv2.getTickCount()
    
    frame_blob = cv2.dnn.blobFromImage(cv2.resize(image_coding, RESIZED_DIMENSIONS), 
                     IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)
    
    neural_network.setInput(frame_blob)

    neural_network_output = neural_network.forward()
    
    for i in np.arange(0, neural_network_output.shape[2]):
        confidence = neural_network_output[0, 0, i, 2]
        if confidence > 0.30:
            idx = int(neural_network_output[0, 0, i, 1])
            detected_category = classes[idx]
            print(detected_category)
            try:
                if detected_category == 'person':
                    bounding_box = neural_network_output[0, 0, i, 3:7] * np.array([width*0.5, height*0.6, width*0.6, height*0.6])
                    (startX, startY, endX, endY) = bounding_box.astype("int")
                    bounding_box = (startX, startY, endX, endY)
                    
                    cv2.rectangle(frame, bounding_box, (0,0,255),2)
                    cv2.putText(frame, detected_category, (bounding_box[0]+10, bounding_box[1]+40), font, fontScale=font_scale, color=(0,255,0), thickness =3)
                    mid_point_person = (int(int(bounding_box[0] + int(bounding_box[2])/2)), int(int(bounding_box[1] + int(bounding_box[3])/2)))
                    cv2.circle(frame, mid_point_person , 10, (0, 0, 255), 2)
            except IndexError:
                continue


    # Update tracker
    # ok_1, bbox_1 = tracker_1.update(frame)
    # ok_2, bbox_2 = tracker_2.update(frame)
    ok_3, bbox_3 = tracker_3.update(frame)
    ok_4, bbox_4 = tracker_4.update(image_coding)
    # ok_5, bbox_5 = tracker_5.update(frame)
    ok_6, bbox_6 = tracker_6.update(image_coding)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box
    # if ok_1:
    #     # Tracking success
    #     p1 = (int(bbox_1[0]), int(bbox_1[1]))
    #     p2 = (int(bbox_1[0] + bbox_1[2]), int(bbox_1[1] + bbox_1[3]))
    #     cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    #     cv2.putText(frame, "Tracker BOOSTING", p1,
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    # if ok_2:
    #     # Tracking success
    #     p1 = (int(bbox_2[0]), int(bbox_2[1]))
    #     p2 = (int(bbox_2[0] + bbox_2[2]), int(bbox_2[1] + bbox_2[3]))
    #     cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    #     cv2.putText(frame, "Tracker MIL", p1,
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    if ok_3:
        # Tracking success
        p1 = (int(bbox_3[0]), int(bbox_3[1]))
        p2 = (int(bbox_3[0] + bbox_3[2]), int(bbox_3[1] + bbox_3[3]))
        cv2.rectangle(frame, p1, p2, (0, 165, 255), 2, 1)
        cv2.putText(frame, "Tracker KCF", p1,
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        mid_point_kcf = (int(int(bbox_3[0] + int(bbox_3[2])/2)), int(int(bbox_3[1] + int(bbox_3[3])/2)))
        # print(mid_point_kcf)
        cv2.circle(frame, mid_point_kcf , 10, (0, 165, 255), 2)
        
    if ok_4:
        # Tracking success
        p1 = (int(bbox_4[0]), int(bbox_4[1]))
        p2 = (int(bbox_4[0] + bbox_4[2]), int(bbox_4[1] + bbox_4[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        cv2.putText(frame, "Tracker TLD", p1,
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        mid_point_tld = (int(int(bbox_4[0] + int(bbox_4[2])/2)), int(int(bbox_4[1] + int(bbox_4[3])/2)))
        print(mid_point_tld)
        cv2.circle(frame, mid_point_tld , 10, (0, 255, 0), 2)
    # if ok_5:
    #     # Tracking success
    #     p1 = (int(bbox_5[0]), int(bbox_5[1]))
    #     p2 = (int(bbox_5[0] + bbox_5[2]), int(bbox_5[1] + bbox_5[3]))
    #     cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    #     cv2.putText(frame, "Tracker MEDIANFLOW", p1,
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    if ok_6:
        # Tracking success
        p1 = (int(bbox_6[0]), int(bbox_6[1]))
        p2 = (int(bbox_6[0] + bbox_6[2]), int(bbox_6[1] + bbox_6[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.putText(frame, "Tracker CSRT", p1,
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        mid_point_csrt = (int(int(bbox_6[0] + int(bbox_6[2])/2)), int(int(bbox_6[1] + int(bbox_6[3])/2)))
        cv2.circle(frame, mid_point_csrt , 10, (255, 0, 0), 2)

    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    center_diff=10
    if ((mid_point_kcf[0] - mid_point_csrt[0])<center_diff or (mid_point_kcf[1] - mid_point_csrt[1])<center_diff) :
        cv2.circle(frame, mid_point_csrt , 10, (0, 255, 255), 2)
        cv2.putText(frame, "Follow me" , mid_point_csrt ,
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    
    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    
    # Display result
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
