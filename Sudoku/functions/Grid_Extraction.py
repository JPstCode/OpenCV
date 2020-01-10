import cv2 as cv
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import math

def takeSecond(hor):
    return hor[1]

def takeFirst(ver):
    return ver[0]



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
        if scale > 0.92 and scale < 1.08:
            grid.append(contour[:,0])
            minmax.append((min_x,min_y))
            minmax.append((max_x,max_y))
            break

    if len(grid) == 0:
        return [],0

    else:
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

    ref_points = ((min_x-5, min_y-10), (min_x-10, max_y+10), (max_x+10,max_y+10), (max_x+10, min_y-10))

    leftop = (0,0)
    lefbot = (0,0)
    rightbot = (0,0)
    righttop = (0,0)



    for corner_number,point in enumerate(ref_points):
        prev_dst = 100

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

                prev_dst = dst

    corners = np.asanyarray((leftop, lefbot, righttop, rightbot))

    #Drav corner points
    for index,cor in enumerate(corners):
        cv.circle(color,ref_points[index],5,(0,0,255),3)
        cv.circle(color,tuple(cor),5,(0,255,0),3)

    plt.figure()
    plt.imshow(color)
    plt.show()


    return corners



def add_frame(img):

    img[:5,:] = 255
    img[:,:5] = 255
    img[-5:,:] = 255
    img[:,-5:] = 255

    return img

def clear_grid(img,hor,ver):

    width = 8
    for line in hor:

        y = line[1]
        y2 = line[3]

        if y < width:
            low = 0
        else:
            low = y - width
        if y2 < width:
            low2 = 0
        else:
            low2 = y2 - width

        img[low:y+width,:] = 0
        img[low2:y2+width,:] = 0

    for line in ver:

        x = line[0]
        x2 = line[2]

        if x < width:
            low = 0
        else:
            low = x - width
        if x2 < width:
            low2 = 0
        else:
            low2 = x2 - width

        img[:,low:x + width] = 0
        img[:,low2:x2 + width] = 0

    return img



# Divide perspective transformed image to cells
def get_cells(img, color):

    #Img size 540x540, 9x9x60 cells

    blurred = cv.GaussianBlur(img, (5, 5),2)
    thres = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,9,2)

    # Add 'frame' to extract all lines
    thres = add_frame(thres)

    linesP = cv.HoughLinesP(thres, 1, np.pi/180, 150, None, 400, 200)

    # Sort to horisontal and vertical lines
    hor = []
    ver = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            dx = abs(l[0] - l[2])
            dy = abs(l[1] - l[3])
            if dx < dy:
                ver.append(l)
            else:
                hor.append(l)

    # Remove duplicates
    hor.sort(key=takeSecond)
    ver.sort(key=takeFirst)

    cleared_hor = []
    cleared_ver = []

    prev_coord = 0
    for l in hor:

        if len(cleared_hor) == 0:
            cleared_hor.append(l)
        else:
            dy = abs(l[1] - prev_coord)

            if dy < 10:
                continue
            else:
                cleared_hor.append(l)

        prev_coord = l[1]

    for l in ver:

        if len(cleared_ver) == 0:
            cleared_ver.append(l)
        else:
            dy = abs(l[0] - prev_coord)

            if dy < 10:
                continue
            else:
                cleared_ver.append(l)

        prev_coord = l[0]



    for l in cleared_hor:
        cv.line(color, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    for l in cleared_ver:
        cv.line(color, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_AA)

    thres = clear_grid(thres,cleared_hor,cleared_ver)

    plt.figure(1)
    plt.imshow(thres,'gray')
    plt.show()

    print(len(cleared_hor))
    print(len(cleared_ver))


    #plt.figure(2)
    #plt.imshow(canny,'gray')

    plt.figure(1)
    plt.imshow(color)
    plt.show()


    cells = []
    counter = 0
    for i in range(0,9):
        cells.append([])
        for j in range(0,9):

            x1 = i*60
            x2 = (i+1)*60
            y1 = j*60
            y2 = (j+1)*60

            cell = cv.resize(thres[x1:x2,y1:y2], (20,20),interpolation=cv.INTER_LINEAR_EXACT)
            cell_color = cv.cvtColor(cell,cv.COLOR_GRAY2BGR)

            contours,_ = cv.findContours(cell,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(cell_color, contours, -1, (255, 0, 0), 1)


#            Show individual celss
            print(np.sum(cell))
            plt.figure(1)
            plt.imshow(cell,'gray')

            plt.figure(2)
            plt.imshow(cell_color)
            plt.show()

            if np.sum(cell) > 2000:
                counter = counter + 1
                cells[i].append(cell)

    cells = np.asanyarray(cells)
    print(counter)

    np.save('s6',cells)

    return cells



#4 was off

if __name__ == '__main__':

    path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\sudokus\sudoku4.jpg'

    color = cv.imread(path,1)
    img = cv.cvtColor(color,cv.COLOR_BGR2GRAY)
    blurred = cv.blur(img, (3, 3))
    (rows,cols) = img.shape


    total = rows*cols
    thres_pixels = 0
    threshold = 0

    hist_ratios = np.arange(0.12,0.5,0.03)
    contour_iter = 0
    grid_contour = []
    minmax = []

    while len(grid_contour) == 0:

        hist = cv.calcHist([blurred], [0], None, [256], [0, 256])

        hist_ratio = hist_ratios[contour_iter]
        # threshold
        for iter,pixel in enumerate(hist):
            thres_pixels = thres_pixels + pixel
            if thres_pixels/total >= hist_ratio:
                threshold = iter
                break


        _,thres = cv.threshold(blurred,threshold,255,cv.THRESH_BINARY_INV)

        # Find grid contours
        contours,_ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


        big_contours = []
        for contour in contours:
            #print(cv.contourArea((contour)))
            if cv.contourArea(contour) > 100000:

                # cv.drawContours(color, [contour], -1, (255, 0, 0), 3)
                # plt.figure()
                # plt.imshow(color)
                # plt.show()

                big_contours.append(contour)


        grid_contour, minmax = grid_check(big_contours)

        contour_iter = contour_iter + 1
        thres_pixels = 0
        threshold = 0

    # Give every 3rd coordinate
    cv.drawContours(color, grid_contour, -1, (255, 0, 0), 3)

    corner = grid_corners(grid_contour,minmax)
    orig_corner = np.asanyarray(((0,0), (0, 540), (540, 0), (540,540)))

    M = cv.getPerspectiveTransform(np.float32(corner), np.float32(orig_corner))
    perspective = cv.warpPerspective(img, M, (540, 540))

    perspective_color = cv.warpPerspective(color,M,(540,540))
    cells = get_cells(perspective,perspective_color)

    # Give every 3rd coordinate
    #cv.drawContours(color, grid_contour, -1, (255, 0, 0), 3)

    #x = np.arange(len(grid_contour[0]))
    # plt.figure(2)
    # plt.imshow(perspective)
    # plt.figure(1)
    # plt.imshow()
    # plt.show()


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

