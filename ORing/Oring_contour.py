import numpy as np
import cv2 as cv
import glob

off = []
images = [cv.imread(file,1) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\ORing\Training\*.jpg')]


for img in images:

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.bilateralFilter(gray,9,220,55)
    _,thres = cv.threshold(blur,100,255,cv.THRESH_BINARY_INV)
    _,contour,hierarchy = cv.findContours(thres,cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)

    index = 0
    cnt = contour[0]
    if cv.contourArea(cnt) < 1000:
        index = 1
        cnt = contour[1]


    cv.drawContours(img,contour,index,(0,255,0),2)
    (x,y), radius = cv.minEnclosingCircle(cnt)
    #cv.circle(thres, (int(x),int(y)),int(radius-2),(255),1)

    center = (int(x),int(y))
    resolution = 400
    angle = np.arange(0, 2*np.pi + (2*np.pi/resolution),
                          2*np.pi/resolution)

    for i in range(0,resolution):

        xx = int(np.cos(angle[i])*0.95*radius+center[1])
        xy = int(np.sin(angle[i])*0.95*radius+center[0])

            #img[xx,xy] = [0,255,0]

        if thres[xx,xy] < 200:

            if np.sum(img[xx,xy]) != 255:
                img[xx,xy] = [0,0,255]

                off.append([xx,xy])

    if len(off) > 2:
        print("Damaged")
        midpoint = int(len(off)/2)
        off_center = (int(off[midpoint][1]),int(off[midpoint][0]))
        mark = cv.circle(img,off_center,10,(0,0,255),1)

    else:
        print("Good")

    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    off.clear()