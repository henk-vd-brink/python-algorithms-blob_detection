import cv2
import numpy as np

cap = cv2.VideoCapture(0)

print(cap)

while True:
    ret, frame = cap.read()
    # print(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
