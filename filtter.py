import numpy as np
import cv2
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
