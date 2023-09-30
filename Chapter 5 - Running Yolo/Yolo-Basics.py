from ultralytics import YOLO
import cv2 # For keyboard binding function, waitKey(0)

model = YOLO('yolov8n.pt')

# yolov8n -> nano version of yolov8


# .pt uzantısı nedir ?
"""
PyTorch checkpoint files with the ".pt" extension to save the learned weights and parameters of a trained model.
"""

results = model('Images/2.png',show=True)

cv2.waitKey(0) # Unless user inputs, do NOT do anything !

"""
from ultralytics import YOLO

# Create an instance of YOLO model
yolo = YOLO()

# Load the pre-trained YOLO model weights
yolo.load("yolov5s.pt")

# Perform inference on an image
result = yolo("path/to/image.jpg")

# Display the detected results
result.show()

# Save the results to a new image file
result.save("path/to/output.jpg")

# Perform inference on a video
yolo.video("path/to/input_video.mp4", "path/to/output_video.mp4")

# Perform inference on a live webcam stream
yolo.stream()

# You can also perform batch inference on multiple images
images = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
results = yolo(images)

# Access detected objects and their attributes
for det in results.pred:
    print(det)

# Additional options can be used to customize the inference behavior
yolo.conf = 0.5  # Set the confidence threshold
yolo.iou = 0.5   # Set the intersection over union threshold

"""
