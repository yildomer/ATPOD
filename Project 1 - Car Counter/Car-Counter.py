import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math


from sort import *

#create instance of SORT
tracker = Sort(max_age=10, min_hits=2,iou_threshold=0.3)
# max_age -> Wait for it to come back

limits = [400,297,673,297] # Limits of line

import torch
# print(torch.backends.mps.is_available())

capture = cv2.VideoCapture("../Videos/cars.mp4")

# capture.set(3,1280) # Width
# capture.set(4,720) # Height
# Check for other propId values and corresponding features.
# For videos, no need to arrange 'width' and 'height'.

model = YOLO('yolov8l.pt')
names = model.names

output_video = cv2.VideoWriter('car-counted.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1280,720))

mask = cv2.imread("mask.png")


counted_ids = []

# Reading an PNG file as it is shown in media viewer.
imageGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED) # Why reading it in while loop ?
"""
    # IMREAD_UNCHANGED: This flag is used to load the image exactly as it is in the file,
    # without any modification or conversion. It's particularly useful when you want to load
    # images that have multiple channels (e.g., RGBA images with alpha channels) and 
    # you want to preserve all the channels without any alteration.
"""

while True:
    success, image = capture.read()

    # Masking
    imageRegion = cv2.bitwise_and(image,mask)

    # Inserting
    image = cvzone.overlayPNG(image,imageGraphics,(0,0))

    # results = model(image, stream = True, device = "mps")
    results = model(imageRegion,stream=True,device = "mps")
    # stream -> efficient use of generators
    # device -> 'mps' -> GPU backend

    detections = np.empty((0,5))
    """
    The code np.empty((0, 5)) creates an empty NumPy array
    with 0 rows and 5 columns. It might seem a bit confusing 
    at first, but it's a way to create an empty array that 
    you can later append rows to. When you print this array, 
    you won't see any values because it's initialized with 
    uninitialized memory. 
    """

    cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),color = (0,0,255), thickness = 3)

    for result in results:

        boxes = result.boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            w,h = x2-x1,y2-y1
            bbox = x1,y1,int(w),int(h)

            confidence = box.conf[0]
            confidence = math.ceil(confidence*100) / 100

            class_index = int(box.cls[0])

            current_class = names[class_index]

            if (current_class in ["car","bus","motorcycle","truck"]) \
                    and (confidence > 0.3):

                cvzone.cornerRect(image, bbox, l=10,rt=5) # rt: rectangle thickness

                cvzone.putTextRect(image,f'{current_class} {confidence}',(max(0,x1),max(20,y1-5)),scale = 1, thickness = 1,offset = 3)

                current_array = np.array([x1,y1,x2,y2,confidence])
                detections = np.vstack((detections,current_array)) # Adding box to detections.

    resultsTracker = tracker.update(dets=detections)

    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        # print(id)
        x1, y1, x2, y2, id = int(x1),int(y1),int(x2),int(y2),int(id)
        w,h = x2-x1, y2-y1
        cvzone.cornerRect(image,(x1,y1,w,h), l=10, rt = 2, colorR=(255,0,0))
        cvzone.putTextRect(image, f'ID {id}', (max(0, x1), max(0, y1 - 20)), scale=1.3, thickness=1, offset=1,colorR=(0,255,0),colorT=(0,0,0))

        cx,cy = x1+w//2, y1+h//2

        cv2.circle(image,(cx,cy),radius=3,color = (255,255,255),thickness=cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[3] + 20:
            if not id in counted_ids:
            # if counted_ids.count(id) == 0: # (Alternative)
                counted_ids.append(id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=3)

    # Displaying counter.
    # cvzone.putTextRect(image,f'Count: {len(counted_ids)}',(50,50),1,1) # Show counter.
    cv2.putText(image,f'{len(counted_ids)}',(220,100),cv2.FONT_HERSHEY_PLAIN,5,color=(0,0,0),thickness=10)

    output_video.write(image)

    cv2.imshow('Video',image)
    # cv2.imshow('ImageRegion',imageRegion)
    cv2.waitKey(1)

# Closes all the frames
cv2.destroyAllWindows()