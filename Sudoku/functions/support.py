import numpy as np
from functions.vision import *

def grid_contour(contour):

    for i in range(0, len(contour)):
        if cv.contourArea(contour[i]) > 15000:
           return contour[i], int(i)


"""Takes grid contour points and returns the corner points"""
def corner_coordinates(cnt):

    coords = []
    prev_x = 0
    prev_y = 0
    prev_dx = 0
    prev_dy = 0
    for c in cnt:

        dx = prev_x - c[0][0]
        dy = prev_y - c[0][1]

        if np.abs(dx) > 5 or np.abs(dy) > 5:

            if np.abs(prev_dx-dx) != 0 and np.abs(prev_dy-dy) != 0:
                #cv.circle(img, (prev_x, prev_y), 5, (0, 0, 255), -1)
                #cv.imshow('img',img)
                #cv.waitKey(0)
                if prev_x != 0 and prev_y != 0:
                    coords.append([prev_x,prev_y])

            prev_dx = dx
            prev_dy = dy

        prev_x = c[0][0]
        prev_y = c[0][1]

    coords = np.asarray(coords, dtype='float32')

    return coords

def get_image_cornes(img):

    size = len(img)
    return np.float32([[0, 0], [0, size], [size, size], [size, 0]])

def make_grid(img):

    grid = []
    box = int(len(img)/9)
    intend = 15
    for i in range(0,9):
        grid.append([])
        for j in range(0,9):
            grid[i].append([])

            x1 = i*box
            x2 = i*box + box
            y1 = j*box
            y2 = j*box + box

            roi = img[x1:x2, y1:y2]
            number = roi[intend: (box - intend), intend: (box - intend)]
            if np.sum(number) > 50000:
                number = resize(number, 28)
                number = gaussian_blur(number, 3)
                grid[i][j].append(number)

    grid = np.asarray(grid)
    return grid


