from ultralytics import YOLO
import cv2
import cvzone # Display detection easier.
import math
import numpy as np

limits = [584,228,718,880] # Limits of line
counted_ids = []
from sort import *

#create instance of SORT
tracker = Sort(max_age=20, min_hits=2,iou_threshold=0.3)
# max_age -> Wait for it to come back

cap = cv2.VideoCapture('uskudar.mp4')
cap.set(3,960) # Width
cap.set(4, 540) # Height

# What is propId and value for set ?

model = YOLO('yolov8l.pt')
names = model.names

mask = cv2.imread("mask.png")

output_video = cv2.VideoWriter('uskudar-count.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                          10, (1920,1080))

# Reading an PNG file as it is shown in media viewer.
imageGraphics = cv2.imread("rsz_atp.png",cv2.IMREAD_UNCHANGED) # Why reading it in while loop ?
"""
    # IMREAD_UNCHANGED: This flag is used to load the image exactly as it is in the file,
    # without any modification or conversion. It's particularly useful when you want to load
    # images that have multiple channels (e.g., RGBA images with alpha channels) and 
    # you want to preserve all the channels without any alteration.
"""

while True:
    success, image = cap.read()

    cv2.resize(image,(960,540))
    cv2.resize(mask,(960,540))

    imageRegion = cv2.bitwise_and(image,mask)

    # cvzone.overlayPNG(image, imageGraphics, (800, 400))

    results = model(imageRegion, stream = True, device="mps") # stream = True for efficient use of generators.

    detections = np.empty((0, 5))
    """
    The code np.empty((0, 5)) creates an empty NumPy array
    with 0 rows and 5 columns. It might seem a bit confusing 
    at first, but it's a way to create an empty array that 
    you can later append rows to. When you print this array, 
    you won't see any values because it's initialized with 
    uninitialized memory. 
    """

    cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 0, 255), thickness=3)

    for r in results:

        boxes = r.boxes

        for box in boxes:

            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            # print(type(box.xyxy[0])) # <class 'torch.Tensor'>
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(image,(x1,y1),(x2,y2), color =(255,0,255), thickness=3,) # color: BGR

            # x1, y1, w, h = box.xywh[0]

            w, h = x2-x1, y2-y1

            bbox = int(x1), int(y1), int(w), int(h)

            # cvzone.cornerRect(image,bbox)
            # cv2.rectangle(image,(x1,y1),(x2,y2),(0,100,100),3)

            # Confidence
            confidence = box.conf[0] # returns confidence value
            confidence = math.ceil((confidence*100))/100 # ceil(x100)/100 -> two decimal places
            # print(confidence)

            #cv2.putText(image, f'{confidence}', (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX)

            # Class Name
            class_index = int(box.cls[0])
            current_class = names[class_index]

            if (current_class in ["person"]) \
                    and (confidence > 0.3):

                cvzone.cornerRect(image, bbox, l=10,rt=5) # rt: rectangle thickness

                cvzone.putTextRect(image,f'{current_class} {confidence}',(max(0,x1),max(20,y1-5)),scale = 1, thickness = 1,offset = 3)

                current_array = np.array([x1,y1,x2,y2,confidence])
                detections = np.vstack((detections,current_array)) # Adding box to detections.

            # cvzone.putTextRect(image, f'{names[int(cls)]} {confidence}', (max(0, x1), max(35, y1-20)), scale = 2, thickness=2)

    resultsTracker = tracker.update(dets=detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        # print(id)
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(image, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(image, f'ID {id}', (max(0, x1), max(0, y1 - 20)), scale=1.3, thickness=1, offset=1,
                           colorR=(0, 255, 0), colorT=(0, 0, 0))

        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(image, (cx, cy), radius=3, color=(255, 255, 255), thickness=cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] < cy < limits[3]:
            if not id in counted_ids:
                # if counted_ids.count(id) == 0: # (Alternative)
                counted_ids.append(id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=3)

        # Displaying counter.
        # cvzone.putTextRect(image,f'Count: {len(counted_ids)}',(50,50),1,1) # Show counter.
        cv2.putText(image, f'Passed Person Count: {len(counted_ids)}', (100, 100), cv2.FONT_HERSHEY_PLAIN, 5, color=(255, 0, 255), thickness=10)

        output_video.write(image)

        cv2.imshow('Video', image)
        # cv2.imshow('ImageRegion',imageRegion)
        cv2.waitKey(1)

cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()