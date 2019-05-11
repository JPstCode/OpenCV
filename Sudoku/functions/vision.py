import cv2 as cv
import glob

def resize(img,size):
    return cv.resize(img, (size, size))

def perspective_transform(img, orig_points, img_corner_points):
    M = cv.getPerspectiveTransform(orig_points, img_corner_points)
    size = len(img)
    return cv.warpPerspective(img, M, (size, size))



def gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def blur(img, kernel, iterations):

    if iterations > 1:
        blur = cv.blur(img, (kernel, kernel))
        for i in range(0, iterations):
            blur = cv.blur(blur, (kernel, kernel))
        return blur

    return cv.blur(img, (kernel, kernel))

def gaussian_blur(img, kernel):
    return cv.GaussianBlur(img,(kernel, kernel), 0)

def threshold(img, min, max, inverse):

    if inverse == False:
        return cv.threshold(img, min, max, cv.THRESH_BINARY)
    else:
        return cv.threshold(img, min, max, cv.THRESH_BINARY_INV)

def adaptive_thres(img, gaussian):
#    return cv.adaptiveThreshold(img, 255, 0, 1, 5, 2)

    if gaussian == True:
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY_INV, 5, 2)
    else:
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY_INV, 5, 2)

def otsu_thres(img):

    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

def get_contours(img):

    _, contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(img, contours, index, rgb):

    if rgb == 0:
        color = (0, 0, 255)
    elif rgb == 1:
        color = (0, 255, 0)
    elif rgb == 2:
        color = (255, 0, 0)
    else:
        color = (255, 255, 0)


    return cv.drawContours(img, contours, index, color, 2)


def show_image_name(name, img):

    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def show_image(*args):

    i = 0
    for arg in args:
        cv.imshow('img'+str(i), arg)
        i+=1

    cv.waitKey(0)
    cv.destroyAllWindows()


def get_images(type):
    images = [cv.imread(file, type) for file in glob.glob(
        r'C:\Users\juhop\Python_Files\Sudoku\sudoku*.png')]
    return images