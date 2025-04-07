import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO

from sort import *

# video_path = r'/Users/kminseok/_dev/tanzu-main/_experiments/video_analysis/carsvid.mp4'
video_path = r'/Users/kminseok/_dev/tanzu-main/_experiments/video_analysis/video_2025-04-07-15-32-03.mp4'

cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8m.pt')

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()
#road_zoneA = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
road_zoneA = np.array([[30, 1050], [700, 1050], [700, 710], [30, 710], [30, 1050]], np.int32)
zoneA_Line = np.array([road_zoneA[0],road_zoneA[1],road_zoneA[2],road_zoneA[3]]).reshape(-1)

print (" zoneA_Line[0,1]",zoneA_Line[0], zoneA_Line[1], "zoneA_Line[2,3]:",zoneA_Line[2],zoneA_Line[3],"zoneA_Line[4,5]:",zoneA_Line[4],zoneA_Line[5],"zoneA_Line[6,7]:",zoneA_Line[6],zoneA_Line[7])

tracker = Sort()
zoneAcounter = []
frameRate = 33
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920,1080))
    results = model(frame, verbose=False)
    current_detections = np.empty([0,5])

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if class_detect == 'car' or class_detect == 'truck' or class_detect == 'bus' and conf > 60:
                detections = np.array([x1,y1,x2,y2,conf])
                current_detections = np.vstack([current_detections,detections])

    cv2.polylines(frame,[road_zoneA], isClosed=False, color=(0, 0, 255), thickness=8)

    track_results = tracker.update(current_detections)
    for result in track_results:
        x1,y1,x2,y2,id = result
        x1,y1,x2,y2,id = int(x1),int(y1),int(x2),int(y2),int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 

        # print (zoneA_Line[5] < cy < zoneA_Line[3] ,  zoneA_Line[0] < cx < zoneA_Line[3],  zoneA_Line[5], cy , zoneA_Line[5], "and " , zoneA_Line[0] , cx , zoneA_Line[3])
        if zoneA_Line[5] < cy < zoneA_Line[3]  and zoneA_Line[0] < cx < zoneA_Line[3]:
            if zoneAcounter.count(id) == 0:
                print ( id)
                zoneAcounter.append(id)
        cv2.circle(frame,(970,90),15,(0,0,255),-1)

        cvzone.putTextRect(frame, f'LANE A Vehicles ={len(zoneAcounter)}', [50, 400], thickness=4, scale=2.3, border=2)

    cv2.imshow('frame', frame)
    cv2.waitKey(60)

cap.release()
cv2.destroyAllWindows()
