from ultralytics import YOLO
import cv2
import cvzone # Display detection easier.
import math

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('../Videos/bikes.mp4')

# cap.set(3,1280) # Width
# cap.set(4, 720) # Height

# What is propId and value for set ?

model = YOLO('yolov8l.pt')
names = model.names


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]



while True:
    success, image = cap.read()

    results = model(image, stream = True) # stream = True for efficient use of generators.

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

            cvzone.cornerRect(image,bbox)
            # cv2.rectangle(image,(x1,y1),(x2,y2),(0,100,100),3)

            # Confidence
            confidence = box.conf[0] # returns confidence value
            confidence = math.ceil((confidence*100))/100 # ceil(x100)/100 -> two decimal places
            # print(confidence)

            #cv2.putText(image, f'{confidence}', (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX)

            # Class Name
            cls = box.cls[0] # Returns an index.

            cvzone.putTextRect(image, f'{names[int(cls)]} {confidence}', (max(0, x1), max(35, y1-20)),scale = 1,thickness=1)

    cv2.imshow('Image',image)

    cv2.waitKey(1)