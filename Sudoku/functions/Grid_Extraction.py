import cv2 as cv
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\sudokus\sudoku3.jpg'

#Check that the received contour matches with grid features
def grid_check(contours):
    """ :return Grid contour
        :return Min and max for x and y
    """

    grid = []
    minmax = []
    for contour in contours:

        x_coords = contour[:,0,0]
        y_coords = contour[:,0,1]
        min_y = np.amin(y_coords)
        max_y = np.amax(y_coords)
        min_x = np.amin(x_coords)
        max_x = np.amax(x_coords)


        scale = (max_x - min_x) / (max_y - min_y)
        if scale > 0.90 and scale < 1.10:
            grid.append(contour[:,0])
            minmax.append((min_x,min_y))
            minmax.append((max_x,max_y))


    return np.asanyarray(grid), np.asanyarray(minmax)


# def histogram_equalization(img):
#
#     hist, bins = np.histogram(img.flatten(), 256, [0, 256])
#
#     cdf = hist.cumsum()
#     cdf_m = np.ma.masked_equal(cdf, 0)
#     cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
#     cdf = np.ma.filled(cdf_m, 0).astype('uint8')
#
#     img2 = cdf[img]
#
#     return img2


def grid_corners(cnt, minmax):

    min_x = minmax[0][0]
    min_y = minmax[0][1]
    max_x = minmax[1][0]
    max_y = minmax[1][1]

    ref_points = ((min_x-10, min_y-10), (min_x-10, max_y+10), (max_x+10,max_y+10), (max_x+10, min_y-10))

    leftop = (0,0)
    lefbot = (0,0)
    rightbot = (0,0)
    righttop = (0,0)


    counter = 0
    for corner_number,point in enumerate(ref_points):
        prev_dst = 50
        close_flag = False

        for cnt_point in cnt[0]:
            dst = distance.euclidean(cnt_point, point)
            if dst < prev_dst:

                if corner_number == 0:
                    leftop = cnt_point

                elif corner_number == 1:
                    lefbot = cnt_point

                elif corner_number == 2:
                    rightbot = cnt_point

                elif corner_number == 3:
                    righttop = cnt_point

                counter = 0
                close_flag = True
                prev_dst = dst

            elif dst > prev_dst:
                counter = counter + 1

            if counter > 5 and close_flag:
                break

    return np.asanyarray((leftop, lefbot, rightbot, righttop))

if __name__ == '__main__':

    color = cv.imread(path,1)
    img = cv.cvtColor(color,cv.COLOR_BGR2GRAY)
    (rows,cols) = img.shape

    blurred = cv.blur(img, (3, 3))
    hist = cv.calcHist([blurred], [0], None, [256], [0, 256])

    total = rows*cols
    thres_pixels = 0
    threshold = 0

    # threshold
    for iter,pixel in enumerate(hist):
        thres_pixels = thres_pixels + pixel
        if thres_pixels/total >= 0.45:
            threshold = iter
            break

    _,thres = cv.threshold(blurred,threshold,255,cv.THRESH_BINARY_INV)

    # Find grid contours
    contours,_ = cv.findContours(thres, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    big_contours = []
    for contour in contours:
        if cv.contourArea(contour) > 100000:
            big_contours.append(contour)


    grid_contour, minmax = grid_check(big_contours)
    corner = grid_corners(grid_contour,minmax)


    # Give every 3rd coordinate
    #cv.drawContours(color, grid_contour, -1, (255, 0, 0), 3)

    #x = np.arange(len(grid_contour[0]))
    #plt.figure(2)
    #plt.plot(x,grid_contour[0])
    #plt.show()


    #plt.figure()
    #plt.hist(area_hist,10)
    #plt.show()


    # plt.figure()
    # plt.imshow(color)
    # plt.show()

    # plt.figure()
    # plt.imshow(thres,'gray')
    # plt.figure()
    # plt.imshow(blurred,'gray')
    #plt.show()

