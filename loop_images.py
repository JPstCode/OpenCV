import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#Loop with test images
for num in range(1,19):

    sample = cv.imread('BabyFood-Test'+str(num)+'.JPG')
    copy = sample.copy()

    #Change color to HSV
    hsv = cv.cvtColor(sample,cv.COLOR_BGR2HSV)

    #Set limits for red value
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([1, 255, 255])

    lower_r_2 = np.array([160, 100, 100])
    upper_r_2 = np.array([180, 255, 255])

    #use mask to hide everything else tha red
    mask = cv.inRange(hsv, lower_red, upper_red)
    mask2 = cv.inRange(hsv, lower_r_2, upper_r_2)

    res = cv.bitwise_and(copy, copy, mask=mask)
    res2 = cv.bitwise_and(copy, copy, mask=mask2)

    #combine masks
    img3 = cv.add(res, res2)

    #Create kernel
    kernel = np.ones([5,5],np.uint8)

    #use erosion to erode the picture
    gray_orig = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    erosion = cv.erode(gray_orig,kernel, iterations=1)



    #search for contours
    _, contours, hierarchy = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)


    #Black image
    if len(contours) == 0:
        cv.imshow('img3', img3)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        for i in range(0,len(contours)):
            cnt = contours[i]

            #If red area is big
            if cv.contourArea(cnt) >= 2000:

                #Draw contour to img
                cv.drawContours(img3, contours, i, (0, 0, 255), 3)

                #calculate area, hull and solidity
                area = cv.contourArea(cnt)
                hull = cv.convexHull(cnt)
                hull_area = cv.contourArea(hull)
                solidity = float(area)/hull_area

                #Write to images
                font = cv.FONT_HERSHEY_SIMPLEX

                #Set limits for area and solidity
                if area <= 8000:

                    cv.putText(img3, 'One spoon', (10, 350), font, 1,
                               (255, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('img3', img3)

                elif area > 8000 and area <= 15000 and solidity > 0.4:

                    cv.putText(img3, 'One spoon', (10, 350), font, 1,
                               (255, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('img3', img3)

                elif area > 8000 and area <= 15000 and solidity < 0.4:

                    cv.putText(img3, 'Two spoons', (10, 350), font, 1,
                               (255, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('img3', img3)

                elif area > 15000:

                    cv.putText(img3, 'Two spoons', (10, 350), font, 1,
                               (255, 255, 255), 2, cv.LINE_AA)
                    cv.imshow('img3', img3)




                cv.waitKey(0)
                cv.destroyAllWindows()



