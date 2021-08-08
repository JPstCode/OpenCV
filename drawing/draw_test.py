# https://stackoverflow.com/questions/28340950/opencv-how-to-draw-continously-with-a-mouse

import cv2
import numpy as np

# true if mouse is pressed
drawing = False

# if True, draw rectangle. Press 'm' to toggle to curve

mode = True

radius = 4

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy= x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.circle(img,(x,y),radius,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.circle(img,(x,y),1,(0,0,255),-1)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('Window')
cv2.setMouseCallback('Window',interactive_drawing)


while(1):

    cv2.imshow('Window', img)
    k = cv2.waitKey(1) & 0xFF

    # key: d
    if k == 100:
        if radius - 1 == 0:
            pass
        radius -= 1

    # key: u
    elif k == 117:
        radius += 1

    elif k == 27:
        break

cv2.destroyAllWindows()