import glob
import numpy as np
#from functions import vision, support
from functions.support import *
from functions.vision import *

if __name__ == '__main__':

    #Store images in numpy array
    images = np.asarray(get_images(1))

    #Resize and make grayscale
    img1 = resize(images[0], 900)
    gray_img = gray(img1)

    #Blur
    blur = gaussian_blur(gray_img, 15)
    thres = adaptive_thres(blur, False)

    #Get contours from thersholded image
    contours = get_contours(thres)

    #Get grid contour
    cnt, index = grid_contour(contours)

    #Get grid corner coordinate
    corner_points = corner_coordinates(cnt)

    #Make perpective transform
    img_corner_points = get_image_cornes(img1)
    transformed = perspective_transform(thres, corner_points, img_corner_points)

    grid = make_grid(transformed)



    #show_image(transformed)




