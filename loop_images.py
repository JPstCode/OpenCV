import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


for num in range(1,19):
    sample = cv.imread('BabyFood-Test'+str(num)+'.JPG')
    copy = sample.copy()
    #cv.imshow('img',sample)

    hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([1, 255, 255])

    lower_r_2 = np.array([160, 100, 100])
    upper_r_2 = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower_red, upper_red)
    mask2 = cv.inRange(hsv, lower_r_2, upper_r_2)

    res = cv.bitwise_and(copy, copy, mask=mask)
    res2 = cv.bitwise_and(copy, copy, mask=mask2)

    img3 = cv.add(res, res2)


    #hist = cv.calcHist([img3], [2], None, [256], [0, 256])
    #plt.plot(hist,color = 'r')
    #plt.xlim([0,256])
    #plt.show()


    #dilate
    kernel = np.ones([5,5],np.uint8)
    erosion = cv.erode(img3,kernel, iterations=1)
    #gradient = cv.morphologyEx(img3,cv.MORPH_GRADIENT, kernel)
    gray_orig = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
    gray_erosion = cv.cvtColor(erosion,cv.COLOR_BGR2GRAY)


    #contours
    _, contours, hierarchy = cv.findContours(gray_erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(img3,contours,0,(0,255,0),3)


    if len(contours) == 0:
        cv.imshow('img3', img3)
        print('No Spoon')
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        for i in range(0,len(contours)):
            cnt = contours[i]


            if cv.contourArea(cnt) >= 2000:

                cv.drawContours(img3, contours, i, (0, 255, 0), 3)
                #print(cv.contourArea(cnt))
                #print(cv.isContourConvex(cnt))
                area = cv.contourArea(cnt)
                hull = cv.convexHull(cnt)
                hull_area = cv.contourArea(hull)
                solidity = float(area)/hull_area
                x,y,w,h = cv.boundingRect(cnt)
                rect_area = w*h
                extent = float(area)/rect_area

                if area <= 8000:
                    print("one Spoon")

                elif area > 8000 and area <= 15000 and solidity > 0.4:
                    print("one Spoon")

                elif area > 8000 and area <= 15000 and solidity < 0.4:
                    print("two spoon")

                elif area > 15000:
                    print("two spoon")

                cv.imshow('img3', img3)
                #print('area: ', area)
                #print('solidity: ' , solidity)

                #print('extent: ' , extent)
                #print('')

                cv.waitKey(0)
                cv.destroyAllWindows()

    #cv.imshow('img3',gray_orig)
    #cv.imshow('histogram',hist)


