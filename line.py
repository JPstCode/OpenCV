import cv2 as cv
import numpy as np
import glob

def threshold(image):

    _, thres = cv.threshold(image,254,255,cv.THRESH_BINARY_INV)

    return thres



def draw_contours(thres):

    _, contours, _ = cv.findContours(thres, cv.RETR_TREE,
                                     cv.CHAIN_APPROX_SIMPLE)

    M = cv.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    return contours, cx, cy

def draw_rect(image,contours):
    rect = cv.minAreaRect(contours[0])
    box = cv.boxPoints(rect)

    box = np.int0(box)
    cv.drawContours(image, [box], 0, (0, 0, 255), 1)

    return image,box

#siirrä viiva kulkemaan keskipisteen kautta
def line(image,color_img,contours):

    rows, cols = image.shape[:2]
    [vx, vy, x, y] = cv.fitLine(contours[0], cv.DIST_L2, 0, 0.01, 0.01)

    lefty = int((-x * vy / vx) + y)
    righty = int((cols - x) * vy / vx + y)


    cv.line(color_img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    return [lefty,righty]


def calc_rotation(lefty,righty,shapex):

    rotation = 0
    #print(lefty,righty,shapex)
    if lefty < righty:
        angle = np.rad2deg(np.arctan((righty-lefty)/shapex))
        rotation = 360-angle

    elif lefty == rotation:
        rotation = 0

    else:
        angle = np.rad2deg(np.arctan((lefty-righty)/shapex))
        rotation = angle

    return rotation


images = [cv.imread(file,0) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\Angle_pics\*.png')]

color_images = [cv.imread(file,1) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\Angle_pics\*.png')]

color_img = color_images[3]
image = images[3]

thres = threshold(image)
contours, cx, cy = draw_contours(thres)

image,box = draw_rect(image,contours)

thres_rect = threshold(image)
rect_contours,_,_ = draw_contours(thres_rect)


[lefty,righty] = line(image,color_img,rect_contours)
rotation = calc_rotation(lefty,righty,len(color_img[0]))


cv.putText(color_img,str(int(rotation))+' deg',(10,int(len(color_img[0]))-10)
           ,cv.FONT_HERSHEY_PLAIN,1,(0,0,255),1,cv.LINE_AA)

cv.imshow('image',color_img)
cv.waitKey(0)
cv.destroyAllWindows()