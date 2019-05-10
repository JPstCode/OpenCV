import cv2 as cv
import glob

def resize(img,size):
    return cv.resize(img, (size, size))

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

def adaptive_thres(img, kernel, gaussian, constant):

    if gaussian == True:
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY, kernel, constant)
    else:
        return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                    cv.THRESH_BINARY, kernel, constant)

def otsu_thres(img):
    return cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

def contours(img):
    return cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

def show_image(name, img):

    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def get_images():
    images = [cv.imread(file, 1) for file in glob.glob(
        r'C:\Users\juhop\Python_Files\Sudoku\sudoku*.png')]
    return images