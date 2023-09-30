from ultralytics import YOLO
import cv2
import cvzone # Display detection easier.
import math
import os
import numpy as np

frame_count = 0

# Name of Video (Security Camera Fooatage)
cap = cv2.VideoCapture('P4_422.MOV')
# cap.set(3,960) # Width
# cap.set(4, 540) # Height

# What is propId and value for set ?

model = YOLO('./Yolo-Weights/yolov8l.pt')
names = model.names


mask = cv2.imread("./Masks/mask-p4.png")

directory = "./SavedFrames"
os.chdir(directory)



while True:
    # success, image = cap.read()
    # For checking in one of consecutive 3 frames:
    for skipFrame in range(3):
        success, image = cap.read()

    # cv2.resize(image,(1920,1088))
    # cv2.resize(mask,(1920,1088))

    imageRegion = cv2.bitwise_and(image,mask)

    results = model(imageRegion, stream = True, device="mps") # stream = True for efficient use of generators.
    # results = model(image, stream=True, device="mps")  # stream = True for efficient use of generators.

    for r in results:

        boxes = r.boxes

        for box in boxes:

            """  
            # Bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            w, h = x2-x1, y2-y1
            bbox = int(x1), int(y1), int(w), int(h)

            # cvzone.cornerRect(image,bbox)
            # cv2.rectangle(image,(x1,y1),(x2,y2),(0,100,100),3)
            """

            # Confidence
            confidence = box.conf[0] # returns confidence value
            confidence = math.ceil((confidence*100))/100 # ceil(x100)/100 -> two decimal places


            # Class Name
            class_index = int(box.cls[0])
            current_class = names[class_index]

            if (current_class in ["car","truck","motorcycle","bus"]) \
                    and (confidence > 0.35):
                frame_count += 1
                file_name = "frame" + str(frame_count) + ".jpg"
                cv2.imwrite(file_name,image)

    # cv2.imshow('Image',image)
    # cv2.imshow('Masked',imageRegion)

    # cv2.waitKey(0)

cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()