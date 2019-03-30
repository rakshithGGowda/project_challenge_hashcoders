import numpy as np
import cv2
import imutils
from sam.colorlabeler import ColorLabeler
def nothing(x):
    # any operation
    pass
 
cap = cv2.VideoCapture(0)
 
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 66, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 134, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 243, 255, nothing)
 
font = cv2.FONT_HERSHEY_COMPLEX
 
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]


    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    cl = ColorLabeler()
    
 
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
 
    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
 
    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
 
    # Contours detection
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        color = cl.label(lab, cnt)
 
        if area > 300:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 2)
 
            if len(approx) == 3:
                cv2.putText(frame, "Triangle"+"   "+str(color), (x, y), font, 1, (0, 0, 0))
            elif len(approx) == 4:
                cv2.putText(frame, "Rectangle"+"   "+str(color), (x, y), font, 1, (0, 0, 0))
            elif len(approx) == 5:
                cv2.putText (frame, "pentagon"+"   "+str(color), (x, y), font, 1,(0,0,0))
            elif len(approx) == 6:
                cv2.putText (frame, "hexagon"+"   "+str(color), (x,y),font, 1,(0,0,0))
            elif 10 < len(approx) < 20:
                cv2.putText(frame, "Circle"+"   "+str(color), (x, y), font, 1, (0, 0, 0))
 
 
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Thresh", thresh)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
