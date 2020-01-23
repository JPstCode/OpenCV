import numpy as np
import cv2 as cv
import csv
#from time import sleep
#from matplotlib import pyplot as plt
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

orig_corners = np.asanyarray(((0, 0), (0, 540), (540, 0), (540, 540)))

# def load_data():
#
#     img_path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\ML\test_files\stest{}.npy'.format(testset)
#     label_path = r'C:\Users\juhop\Documents\Python\OpenCV\Sudoku\ML\test_files\test_label{}.csv'.format(testset)
#
#     labels = []
#     with open(label_path) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for row in csv_reader:
#             for number in row:
#                 labels.append(int(number))
#
#     imgs = np.reshape(np.load(img_path),(-1,45,45,1))
#
#     return imgs, labels

def grid_find(thres):

    corner = []
    # Find grid contours
    contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    big_contours = []
    for contour in contours:
        if cv.contourArea(contour) > 80000:
            big_contours.append(contour)

    grid_contour, minmax = grid_check(big_contours)

    if len(grid_contour) != 0:
        corner = grid_corners(grid_contour, minmax)

    if len(corner) == 4:
        return corner
    else:
        return []


def predict_number(cell_pics):

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

        else:
            final_pred = np.bincount(np.array([pred_4, pred_15, pred_adam])).argmax()
            final_predictions.append(final_pred)

    return final_predictions


def draw_numbers(empty_cells, sudoku, intersections, frame, corner):

    blank = np.zeros((540, 540, 3), dtype='uint8')
    for row, location in empty_cells:
        solved_number = sudoku[row][location]
        intersection_point = np.asanyarray(
            (intersections[row][location][0],
             intersections[row][location][1]))

        num_location = (int(intersection_point[0] + 15), int(intersection_point[1] + 50))

        cv.putText(blank, str(solved_number), num_location, font, 2, (0, 255, 0), 2, cv.FILLED)

    M_Inv = cv.getPerspectiveTransform(np.float32(orig_corners), np.float32(corner))
    perspective_inv = cv.warpPerspective(blank, M_Inv, (1024, 768))

    for i, row in enumerate(frame):
        for j, pixel in enumerate(row):
            pers_green = perspective_inv[i][j][1]
            if pers_green > 0:
                pixel[0] = 0
                pixel[1] = 255
                pixel[2] = 0

    return frame


def start_webcam():

    # Open webcam and set height and width for frame
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

    # Variables for comparing two consequtive sudokus
    previous_positions = [(100,100)]
    prev_sudoku = []

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow('frame',frame)

        # Pre-process frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray,(9,9),0)
        thres = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 2)

        # cv.imshow('thres', thres)
        # Find grid corners
        corners = grid_find(thres)

        # Check that all four corners are found
        if len(corners) == 4:

            # Crop grid for cell extraction
            M = cv.getPerspectiveTransform(np.float32(corners), np.float32(orig_corners))
            perspective = cv.warpPerspective(gray, M, (540, 540))

            cell_pics, cell_pos, intersections, empty_cells = get_cells(perspective, None)

            # If two consequtive sudokus are indentical, no need for another solving
            if cell_pos == previous_positions:

                # Draw solution to frame
                frame = draw_numbers(empty_cells, prev_sudoku, intersections, frame, corners)
                cv.imshow('frame', frame)

            else:

                if len(cell_pics) != 0 and len(cell_pos) != 0:

                    # Classify numbers
                    final_predictions = predict_number(cell_pics)
                    # print(final_predictions)
                    if len(final_predictions) != 0:

                        # Prepare sudoku for solving algorithm
                        sudoku = combine_sudoku(cell_pos, final_predictions)

                        # Use backtracking algorithm to solve puzzle
                        solve(sudoku)

                        if 0 not in sudoku[0] and 0 not in sudoku[1]:

                            # Draw solution to frame
                            frame = draw_numbers(empty_cells, sudoku, intersections, frame, corners)

                            # Store sudoku and positions to compare consequtive sudokus
                            previous_positions = cell_pos
                            prev_sudoku = sudoku

                            cv.imshow('frame', frame)

                    else:
                        pass

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

if __name__ == '__main__':
    main()