import numpy as np
import cv2 as cv
import csv
from time import sleep
from matplotlib import pyplot as plt
from keras.models import load_model

from functions.Grid_Extraction import grid_check
from functions.Grid_Extraction import grid_corners
from functions.Grid_Extraction import get_cells

from functions.solver import combine_sudoku
from functions.solver import solve

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

testset = '4'
#
SGD15 = load_model('SGD15_model.h5')
SGD4 = load_model('SGD4_model.h5')
ADAM = load_model('ADAM_model.h5')
#

font = cv.FONT_HERSHEY_SIMPLEX

def load_data():

    img_path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\ML\test_files\stest{}.npy'.format(testset)
    label_path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\ML\test_files\test_label{}.csv'.format(testset)

    labels = []
    with open(label_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            for number in row:
                labels.append(int(number))

    imgs = np.reshape(np.load(img_path),(-1,45,45,1))

    return imgs, labels

def grid_find(thres):


    # thres_pixels = 0
    # hist_ratios = np.arange(0.09, 0.83, 0.03)
    # hist = cv.calcHist([blurred], [0], None, [256], [0, 256])
    # hist_ratio = hist_ratios[contour_iter]
    #
    # # threshold
    # for iter, pixel in enumerate(hist):
    #     thres_pixels = thres_pixels + pixel
    #     if thres_pixels / total_pixels >= hist_ratio:
    #         threshold = iter
    #         break
    #
    corner = []
    # Find grid contours
    contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    big_contours = []
    for contour in contours:
        #print(cv.contourArea((contour)))
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

    if len(corner) == 4:
        return corner,[]
    else:
        return [],big_contours


def start_webcam():

    # Open webcam
    cap = cv.VideoCapture(0)
    #cap.set(cv.CAP_PROP_FRAME_WIDTH, 720)
    #cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)


    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break



        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray,(9,9),0)

        #_, thres = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY_INV)
        thres = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 2)

        cv.imshow('frame',frame)
        #cv.imshow('thres',thres)

        # cv.imshow('thres', thres)
        corner,big_contours = grid_find(thres)

        # if len(big_contours) != 0:
        #     cv.drawContours(frame, big_contours, -1, (255, 0, 0), 3)
        #     cv.imshow('gray', frame)


        if len(corner) == 4:

            #cv.destroyAllWindows()
            orig_corner = np.asanyarray(((0, 0), (0, 540), (540, 0), (540, 540)))

            M = cv.getPerspectiveTransform(np.float32(corner), np.float32(orig_corner))
            perspective = cv.warpPerspective(gray, M, (540, 540))

            #cv.imshow('huut',perspective)

            #perspective_color = cv.warpPerspective(color, M, (540, 540))
            cell_pics, cell_pos, intersections, empty_cells = get_cells(perspective, None)
            blank = np.zeros((540,540,3),dtype='uint8')
            if len(cell_pics) != 0 and len(cell_pos) != 0:

                predictions_15 = SGD15.predict(cell_pics)
                predictions_4 = SGD4.predict(cell_pics)
                predictions_ADAM = ADAM.predict(cell_pics)

                final_predictions = []
                for i in range(0, len(predictions_4)):

                    pred_15 = np.argmax(predictions_15[i])
                    pred_4 = np.argmax(predictions_4[i])
                    pred_adam = np.argmax(predictions_ADAM[i])

                    if pred_adam != pred_15 and pred_15 != pred_4 and pred_4 != pred_adam:
                        final_predictions = []
                        break
                        # plt.figure(1)
                        # plt.imshow(cell_pics[i, :, :, 0], 'gray')
                        # plt.show()
                        #
                        # number = 0
                        # final_predictions.append(int(number))



                    else:
                        final_pred = np.bincount(np.array([pred_4, pred_15, pred_adam])).argmax()
                        final_predictions.append(final_pred)

                #cap.release()
                #cv.destroyAllWindows()

                if len(final_predictions) != 0:

                    # # Prediction Projection
                    # for index,prediction in enumerate(final_predictions):
                    #     cell_location = cell_pos[index]
                    #
                    #     intersection_point = np.asanyarray(
                    #         (intersections[cell_location[0]][cell_location[1]][0],
                    #         intersections[cell_location[0]][cell_location[1]][1]))
                    #
                    #     location = (int(intersection_point[0]+15),int(intersection_point[1]+50))
                    #
                    #     cv.putText(blank, str(prediction), location, font, 2, (0,255,0), 2, cv.FILLED)

                    sudoku = combine_sudoku(cell_pos, final_predictions)
                    solve(sudoku)

                    for row,location in empty_cells:

                        solved_number = sudoku[row][location]
                        intersection_point = np.asanyarray(
                            (intersections[row][location][0],
                            intersections[row][location][1]))

                        num_location = (int(intersection_point[0]+15),int(intersection_point[1]+50))

                        cv.putText(blank, str(solved_number), num_location, font, 2, (0,255,0), 2, cv.FILLED)


                    M_Inv = cv.getPerspectiveTransform(np.float32(orig_corner), np.float32(corner))
                    perspective_inv = cv.warpPerspective(blank, M_Inv, (640, 480))


                    for i,row in enumerate(frame):
                        for j,pixel in enumerate(row):
                            pers_green = perspective_inv[i][j][1]
                            if pers_green > 0:
                                pixel[0] = 0
                                pixel[1] = 255
                                pixel[2] = 0

                    cv.imshow('frame', frame)
                    # plt.figure(1)
                    # plt.imshow(perspective_inv)
                    #
                    # plt.figure(2)
                    # plt.imshow(frame)
                    #
                    # plt.show()

                else:
                    pass
            #return perspective, cell_pos, cell_pics
            else:
                pass

        else:
            pass

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def main():

    start_webcam()

    # plt.figure(1)
    # plt.imshow(perspective,'gray')
    # plt.show()
    #
    # selection = input("Proceed/Take New Pic/Exit [y/n/q]: ")
    #
    # if selection == 'n':
    #     main()
    # elif selection == 'q':
    #     return
    # elif selection == 'y':
    #
    #     predictions_15 = SGD15.predict(images)
    #     predictions_4 = SGD4.predict(images)
    #     predictions_ADAM = ADAM.predict(images)
    #
    #     labels = []
    #
    #     for i in range(0,len(predictions_4)):
    #
    #         pred_15 = np.argmax(predictions_15[i])
    #         pred_4 = np.argmax(predictions_4[i])
    #         pred_adam = np.argmax(predictions_ADAM[i])
    #
    #         if pred_adam != pred_15 and pred_15 != pred_4 and pred_4 != pred_adam:
    #
    #             plt.figure(1)
    #             plt.imshow(images[i,:,:,0],'gray')
    #             plt.show()
    #
    #             number = input("Define number: ")
    #             labels.append(int(number))
    #
    #         else:
    #             final_pred = np.bincount(np.array([pred_4, pred_15, pred_adam])).argmax()
    #             labels.append(final_pred)
    #
    #
    #     print(labels)
    #
    #
    #     plt.figure(1)
    #     plt.imshow(perspective,'gray')
    #     plt.show()

if __name__ == '__main__':
    main()