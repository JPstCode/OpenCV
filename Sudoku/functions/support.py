import numpy as np
from functions.vision import *

def grid_contour(img):
    img = gray(img)
    blur1 = blur(img,5,1)
    blur2 = gaussian_blur(img,5)

    show_image('blur1',blur1)
    show_image('blur2',blur2)





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
                coords.append([prev_x,prev_y])

            prev_dx = dx
            prev_dy = dy

        prev_x = c[0][0]
        prev_y = c[0][1]

    coords = np.asarray(coords)

    return coords
