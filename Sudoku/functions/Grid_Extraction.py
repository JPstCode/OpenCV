import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\sudokus\sudoku4.jpg'



#Check that the received contour matches with grid features
def grid_check(contour):









if __name__ == '__main__':

    color = cv.imread(path,1)
    img = cv.cvtColor(color,cv.COLOR_BGR2GRAY)

    (rows,cols) = img.shape
    # plt.figure(1)
    # plt.imshow(img,'gray')
    # plt.show()



    bins = np.arange(255)
    hist = np.histogram(img,bins)

    most_pixels = np.argmax(hist[0])

    _,thres = cv.threshold(img,(most_pixels-100),255,cv.THRESH_BINARY_INV)

    blurred = cv.blur(thres, (10,10))

    # Find grid contours
    contours,_ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    areas = []
    big_contours = []
    for contour in contours:
        if cv.contourArea(contour) > 500000:
            big_contours.append(contour)




    #plt.figure()
    #plt.hist(area_hist,10)
    #plt.show()

    #
    #
    #
    # linesP = cv.HoughLinesP(blurred, 1, np.pi / 180, 50, None, cols/3, 10)
    #
    # #filter lines
    # prev_loc = (0,0)
    # grid = []
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #
    #         x_dist = np.abs(l[0]-prev_loc[0])
    #         y_dist = np.abs(l[1]-prev_loc[1])
    #
    #         if x_dist > 100 and y_dist > 100:
    #             grid.append(l)
    #             cv.line(color, (l[0], l[1]), (l[2], l[3]), (0,0,255), 25, cv.LINE_AA)
    #
    #         prev_loc = (l[0], l[1])


    plt.figure()
    plt.imshow(color)
    # plt.figure()
    # plt.imshow(thres,'gray')
    # plt.figure()
    # plt.imshow(blurred,'gray')
    plt.show()

