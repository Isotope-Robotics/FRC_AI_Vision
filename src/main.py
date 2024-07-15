# Base Dependencies
import sys
import time
import math
import logging

# Network Tables Dependencies
from networktables import NetworkTables

# OpenCV Dependencies
import cv2 as cv

# AI Model Dependencies
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG)

# Checks if IP is specified
if len(sys.argv) != 2:
    print("Error: specifying IP to robot")
    exit(0)

# Network Tables Setup    
ip = sys.argv[1]
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("SmartDashboard")

# Network Table Entries
auto_value = sd.getAutoUpdateValue("robotTime", 0)

# Camera Setup
stream = cv.VideoCapture(0)

# AI Setup
# Model
model = YOLO("yolov8n.pt")
# Classes for Detection
classNames = ["note"]



# Main Loop of Program
while (True):    
    success, img = stream.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        
        for box in boxes:
        # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv.imshow('Webcam', img)
    if cv.waitKey(1) == ord('q'):
        break

stream.release()
cv.destroyAllWindows()