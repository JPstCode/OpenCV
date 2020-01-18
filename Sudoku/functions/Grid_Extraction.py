import cv2 as cv
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt

sudoku = '4'


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
        prev_dst = 80

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

    if np.sum(leftop) == 0 or np.sum(lefbot) == 0 or np.sum(rightbot) == 0 or np.sum(righttop) == 0:
        return []
    else:
        corners = np.asanyarray((leftop, lefbot, righttop, rightbot))

        # Drav corner points
        # for index, cor in enumerate(corners):
        #     cv.circle(color, ref_points[index], 5, (0, 0, 255), 3)
        #     cv.circle(color, tuple(cor), 5, (0, 255, 0), 3)
        #
        # plt.figure()
        # plt.imshow(color)
        # plt.show()

        return corners


def add_frame(img):

    img[:5,:] = 255
    img[:,:5] = 255
    img[-5:,:] = 255
    img[:,-5:] = 255

    return img

def extend_lines(hor,ver):

    for line in hor:

        if line[0] < 280:
            line[0] = 0
            line[2] = 540

        else:
            line[0] = 540
            line[2] = 0

    for line in ver:
        if line[1] < 280:
            line[1] = 0
            line[3] = 540

        else:
            line[1] = 540
            line[3] = 0

    return hor, ver

def correct_coordinate(lines):

    if len(lines[0]) == 4:

        for line in lines:
            line[1] = -line[1] + 540
            line[3] = -line[3] + 540

    else:
        for point in lines:
            point[1] = int((-point[1] + 540))
            point[0] = int((point[0]))

    return lines


# Calculate the intersection coordinates of grid lines
def get_intersections(hor,ver):

    intersections = []
    for index, h_line in enumerate(hor):
        k_h = ((h_line[3]) - (h_line[1]))/(h_line[2] - h_line[0])
        intersections.append([])

        for v_line in ver:
            k_v = ((v_line[3])-(v_line[1]))/(v_line[2]-v_line[0])

            if np.isinf(k_v):
                x = v_line[0]
            else:
                x = (k_h*h_line[0] - k_v*v_line[0] + v_line[1] - h_line[1]) / (k_h - k_v)

            if k_h == 0:
                y = h_line[1]
            else:
                y = k_h*x - k_h*h_line[0] + h_line[1]

            intersections[index].append([int(x),int(y)])

    return intersections



# Divide perspective transformed image to cells
def get_cells(img, color):

    #Img size 540x540, 9x9x60 cells

    blurred = cv.GaussianBlur(img, (5, 5),2)
    thres = cv.adaptiveThreshold(blurred,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,9,2)

    # Add 'frame' to extract all lines
    thres = add_frame(thres)
    linesP = cv.HoughLinesP(thres, 1, np.pi/180, 150, None, 400, 200)


    # plt.figure(2)
    # plt.imshow(thres,'gray')
    # plt.show()

    # Sort to horisontal and vertical lines
    hor = []
    ver = []
    try:
        if len(linesP) != 0:
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

                    if dy < 15:
                        continue
                    else:
                        cleared_hor.append(l)

                prev_coord = l[1]

            for l in ver:

                if len(cleared_ver) == 0:
                    cleared_ver.append(l)
                else:
                    dy = abs(l[0] - prev_coord)

                    if dy < 15:
                        continue
                    else:
                        cleared_ver.append(l)

                prev_coord = l[0]

            cleared_hor, cleared_ver = extend_lines(cleared_hor,cleared_ver)

            if len(cleared_hor) != 10 or len(cleared_ver) != 10:
                return [],[],[],[]
            else:

                #plt.figure(1)
                #plt.imshow(thres,'gray')
                #plt.show()


                intersections = get_intersections(cleared_hor, cleared_ver)

            cell_pos = []
            cell_pics = []

            for row in range(0,9):
                for coord in range(0,9):
                    x = intersections[row][coord][0]
                    y = intersections[row][coord][1]
                    w = intersections[row][coord+1][0]
                    h = intersections[row + 1][coord][1]
                    cell = thres[y+8:h-8,x+8:w-8]

                    if np.sum(cell) < 20000:
                        continue
                    else:

                        # plt.figure(1)
                        # plt.imshow(cell,'gray')

                        cell = cv.resize(cell, (45,45),interpolation=cv.INTER_NEAREST)

                        # plt.figure(2)
                        # plt.imshow(cell,'gray')
                        # plt.show()


                        #cell_pics[row][coord].append(cell)
                        cell = np.reshape(cell,(45,45,1))
                        cell_pics.append(cell)
                        cell_pos.append((row,coord))



            #train_cells = np.asanyarray(cell_pics)
            #filename = r'stest{}'.format(sudoku)
            #np.save(filename,train_cells)
            if len(cell_pos) != 0 and len(cell_pics) != 0:
                return np.asanyarray(cell_pics), cell_pos, intersections, thres

            else:
                return [],[],[],[]
        else:
            return [], [], [],[]

    except TypeError:
        return [],[],[],[]

if __name__ == '__main__':

    path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\sudokus\pics\test{}.jpg'.format(sudoku)
    color = cv.imread(path,1)
    img = cv.cvtColor(color,cv.COLOR_BGR2GRAY)
    blurred = cv.blur(img, (3, 3))
    (rows,cols) = img.shape


    total = rows*cols
    thres_pixels = 0
    threshold = 0

    hist_ratios = np.arange(0.12,0.8,0.03)
    contour_iter = 0
    corner = []
    grid_contour = []
    minmax = []

    while len(corner) == 0:

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
            # print(cv.contourArea((contour)))
            if cv.contourArea(contour) > 100000:
                #
                # cv.drawContours(color, [contour], -1, (255, 0, 0), 3)
                # plt.figure(1)
                # plt.imshow(thres,'gray')
                #
                # plt.figure(2)
                # plt.imshow(color)
                #
                # plt.show()

                big_contours.append(contour)


        grid_contour, minmax = grid_check(big_contours)

        if len(grid_contour) != 0:
            corner = grid_corners(grid_contour, minmax)

        contour_iter = contour_iter + 1
        thres_pixels = 0
        threshold = 0

    # cv.drawContours(color, grid_contour, -1, (255, 0, 0), 3)

    orig_corner = np.asanyarray(((0,0), (0, 540), (540, 0), (540,540)))

    M = cv.getPerspectiveTransform(np.float32(corner), np.float32(orig_corner))
    perspective = cv.warpPerspective(img, M, (540, 540))

    perspective_color = cv.warpPerspective(color,M,(540,540))
    cell_pics, cell_labels,_ = get_cells(perspective,perspective_color)


    plt.figure(1)
    plt.imshow(perspective, 'gray')
    plt.show()

    print("asd")

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

