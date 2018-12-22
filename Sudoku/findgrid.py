import numpy as np
import cv2 as cv
import glob
from scipy.spatial import distance


def getCoords(cnt):

    xs = []
    ys = []
    cnts = []

    luc = [0, 0]

    for k in cnt:
        xs.append(k[0][0])
        ys.append(k[0][1])
        cnts.append([k[0][0],k[0][1]])

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    for i in range(0,len(xs)):
        if np.sum(np.subtract(cnts[i],[xmin,ymin])) < 10:
            luc = [xmin,ymin]
            ruc = [xmax,ymin]
            rdc = [xmax,ymax]
            ldc = [xmin,ymax]
            coords = np.asarray([luc,ruc,ldc,rdc])


            return coords

    if np.sum(luc) == 0:
        luc = tuple(cnt[cnt[:, :, 0].argmin()][0])
        rdc = tuple(cnt[cnt[:, :, 0].argmax()][0])
        ruc = tuple(cnt[cnt[:, :, 1].argmin()][0])
        ldc = tuple(cnt[cnt[:, :, 1].argmax()][0])
        coords = np.asarray([luc, ruc, ldc, rdc])

        return coords

    return 0,0,0,0


def drawlines(coords,img):

    points1 = []
    points2 = []

    a1 = (coords[1][0] - coords[0][0])
    a2 = (coords[1][1] - coords[0][1])
    b1 = (coords[2][0] - coords[0][0])
    b2 = (coords[2][1] - coords[0][1])

    h1 = (b1/9)
    h2 = (a1/9)
    w1 = (b2/9)
    w2 = (a2/9)

    for i in range(0,10):

        #cv.line(img,(coords[0][0]+int(i*h1),coords[0][1]+int(i*w1)),
        #    (coords[1][0]+int(i*h1),coords[1][1]+int(i*w1)),(0,255,0),2)


        #cv.line(img, (coords[0][0] + int(i * h2), coords[0][1] + int(i * w2)),
        #        (coords[2][0] + int(i * h2), coords[2][1] + int(i * w2)), (255, 0, 0), 2)


        points1.append((coords[0][0] + int(i * h1), coords[0][1] + int(i * w1)))
        points2.append((coords[0][0] + int(i * h2), coords[0][1] + int(i * w2)))

    #cv.imshow('image',img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return points1,points2


def drawRects(points1,points2,r,img):

    print(points1)
    print(points2)
    #input("asd")
    w = int(np.abs(points1[0][0] - points1[1][0]))
    h = int((points2[0][1] - points2[1][1]))
    print(w,h)

    for i in range(0,9):
        cv.rectangle(img, (points1[i][0],points1[i][1]), (points1[i][0]+r,points1[i][1]+r),
                     (0, 0, 255), 2)

        for j in range(1,9):
            cv.rectangle(img,(points2[j][0] + (i*w), points2[j][1] + (i*r)),
                         (points2[j][0] + r + (i*w), points2[j][1] + (i*r) +r), (0,0,255),2)


        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    #        for j in range(0,9):
#            cv.rectangle(img,points1[i],(points1[i] + [int(r),int(r)]),(0,0,255),2)



    input("points")


    return


images = [cv.imread(file,1) for file in glob.glob(
    r'C:\Users\Juho\Python_Files\Sudoku\rect*.png')]

img = cv.resize(images[2],(500,500))
copy = img.copy()
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur = cv.blur(gray,(2,2))

for i in range(0,1):
    blur = cv.blur(blur,(2,2))

_, thres = cv.threshold(blur, 160, 255, cv.THRESH_BINARY_INV)


#cv.imshow('image',thres)
#cv.waitKey(0)
#cv.destroyAllWindows()

_, contours,_ = cv.findContours(thres, cv.RETR_TREE,
                                        cv.CHAIN_APPROX_SIMPLE)

contours = np.asarray(contours)
cnt = contours[0]

for i in range(0,len(contours)):
    if cv.contourArea(contours[i]) > 15000:
        cv.drawContours(copy,contours, i, (0,255,0), 2)
        cnt = contours[i]

coords = getCoords(cnt)

cv.circle(copy, (coords[0][0],coords[0][1]), 5, (0,0,255), -1)
cv.circle(copy, (coords[1][0],coords[1][1]), 5, (0,0,255), -1)
cv.circle(copy, (coords[2][0],coords[2][1]), 5, (0,0,255), -1)
cv.circle(copy, (coords[3][0],coords[3][1]), 5, (0,0,255), -1)

A = cv.contourArea(cnt)
a = np.sqrt(A)
r = a / 9
print(cnt.shape, A, a, r)

#cv.imshow('contours', copy)
#cv.imshow('thres', thres)

#cv.waitKey(0)
#cv.destroyAllWindows()

input("Contour data")

points1, points2 = drawlines(coords,img)
drawRects(points1,points2,int(r),img)








edges = cv.Canny(blur,50,150,apertureSize=3)
minLineLenght = 300
maxLineGap = 70

lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLenght, maxLineGap)
print(lines.shape)

grid_rows = []
grid_columns = []


for line in lines:
    grid_columns.append(line[0][0])
    grid_rows.append(line[0][1])

for i in range(0,len(grid_columns)):

    cv.circle(img, (grid_columns[i],grid_rows[i]), 3, (0, 0, 255), -1)


grid_columns.sort()
grid_rows.sort()

print(grid_rows)
print(grid_columns)
input("hough Data")

#cv.imshow('edges',edges)
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()