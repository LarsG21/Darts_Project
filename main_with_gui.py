import pickle
import sys
from random import randint
from statistics import mode
from time import sleep

import cv2
import numpy as np
from PySide2 import QtCore
from PySide2.QtCore import QThreadPool, QRunnable
from PySide2.QtWidgets import QApplication, QMainWindow

import CalibrationWithUncertainty
import ContourUtils
import DartScore
import dart_scorer_util
import gui
import utils
from FPS import FPS
from qt_ui_classes import DartPositionLabel
from ui_dart_main_gui import Ui_DartScorer

# Globals
points = []
intersectp = []
ellipse_vertices = []
newpoints = []
intersectp_s = []
dart_point = None
ACTIVE_PLAYER = 1

# #############  Config  ####################

loadSavedParameters = True
rows = 6  # 17   6
columns = 9  # 28    9
squareSize = 30  # mm
calibrationRuns = 1
CAMERA_NUMBER = 0  # 0,1 is built-in, 2 is external webcam
TRIANGLE_DETECT_THRESH = 24
useMovingAverage = False
score1 = DartScore.Score(501, True)
score2 = DartScore.Score(501, True)
scored_values = []
scored_mults = []

values_of_round = []
mults_of_round = []

new_dart_point = None
update_dart_point = False

images_for_rolling_average = []
ellipse = None
x_offset_current, y_offset_current = 0, 0

#############################################
STOP_DETECTION = False
dart_id = 0

cap = cv2.VideoCapture(CAMERA_NUMBER)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if loadSavedParameters:
    pickle_in_MTX = open("PickleFiles/mtx_cheap_webcam_good_target.pickle", "rb")
    # pickle_in_MTX = open("PickleFiles/mtx_surface_back.pickle", "rb")
    meanMTX = pickle.load(pickle_in_MTX)
    print(meanMTX)
    pickle_in_DIST = open("PickleFiles/dist _cheap_webcam_good_target.pickle", "rb")
    # pickle_in_DIST = open("PickleFiles/dist_surface_back.pickle", "rb")
    meanDIST = pickle.load(pickle_in_DIST)
    print(meanDIST)
    print("Parameters Loaded")
else:
    meanMTX, meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(cap=cap, rows=rows,
                                                                                                    columns=columns,
                                                                                                    squareSize=squareSize,
                                                                                                    runs=calibrationRuns,
                                                                                                    saveImages=False,
                                                                                                    webcam=True)
target_ROI_size = (600, 600)
resize_for_squish = (600, 600)
previous_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)

default_img = np.zeros(target_ROI_size).astype(np.uint8)


def detect_dart_circle_and_set_limits(img_roi):
    cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = gui.updateTrackBar()
    global contours, radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, cnt, ellipse, x, y, a, b, angle, center_ellipse, x_offset_current, y_offset_current
    imgContours, contours, imgCanny = ContourUtils.get_contours(img=img_roi, cThr=(cannyLow, cannyHigh),
                                                                gaussFilters=noGauss, minArea=minArea,
                                                                epsilon=epsilon, draw=False,
                                                                erosions=erosions, dilations=dilations,
                                                                showFilters=showFilters)
    radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset = gui.update_dart_trackbars()
    # Create Radien in pixels
    for cnt in contours:
        if 200000 / 4 < cnt[1] < 1000000 / 4:
            # Create the outermost Circle
            if ellipse is None or x_offset_current != x_offset or y_offset_current != y_offset:  # Save the outer most ellipse for later to avoid useless re calculation !
                x_offset_current, y_offset_current = x_offset, y_offset
                ellipse = cv2.fitEllipse(cnt[4])  # Also a benefit for stability of the outer ellipse --> not jumping from frame to frame
                x, y = ellipse[0]
                a, b = ellipse[1]
                angle = ellipse[2]
                center_ellipse = (int(x + x_offset / 10), int(y + y_offset / 10))
                a = a / 2
                b = b / 2
                # set the limits also only once
                dart_scorer_util.bullsLimit = a * (radius_1 / 100)
                dart_scorer_util.singleBullsLimit = a * (radius_2 / 100)
                dart_scorer_util.innerTripleLimit = a * (radius_3 / 100)
                dart_scorer_util.outerTripleLimit = a * (radius_4 / 100)
                dart_scorer_util.innerDoubleLimit = a * (radius_5 / 100)
                dart_scorer_util.outerBoardLimit = a * (radius_6 / 100)


class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_DartScorer()
        self.ui.setupUi(self)  # Set up the external generated ui

        # Buttons
        self.ui.set_default_img_button.clicked.connect(lambda: UIFunctions.set_default_image(self))
        self.ui.start_measuring_button.clicked.connect(lambda: UIFunctions.start_detection_and_scoring(self))
        self.ui.stop_measuring_button.clicked.connect(lambda: UIFunctions.stop_detection_and_scoring(self))
        self.ui.undo_last_throw_button.clicked.connect(lambda: UIFunctions.undo_last_throw(self))

        self.DartPositions = {}

        #HIGHLIGHT: NEEDS TO CHANGE ONE LINE in ui_dart_main.py to work !
        # self.ui.dart_board_image = DartPositionLabel(self.ui.Dart_Board_Bg)
        # Changes Type of Label to DartPositionLabel to enable adding of Dart Positions

        # DartPositionId = randint(1, 10000)
        # self.DartPositions[DartPositionId] = DartPositionLabel(self.ui.dart_board_image)
        # self.DartPositions[DartPositionId].addDartPositionRandomly()
        # DartPositionId = randint(1, 10000)
        # self.DartPositions[DartPositionId] = DartPositionLabel(self.ui.dart_board_image)
        # self.DartPositions[DartPositionId].addDartPosition(300,100)

        self.show()


class DefaultImageSetter(QRunnable):
    def closeEvent(self, event):
        super(QRunnable, self).closeEvent(event)
        self.ser.close()

    def __init__(self):
        super().__init__()

    def run(self):
        global default_img
        while True:
            success, img = cap.read()
            if success:
                img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
                cv2.putText(img_undist, "Press x to take and choose an image shown in 'Default' as default", (5, 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
                cv2.imshow("Preview", img_undist)
                img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False)
                if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                    img_roi = cv2.resize(img_roi, resize_for_squish)
                    default_img = img_roi
                    print("Set default image")
                    cv2.imshow("Default", default_img)
                    cv2.waitKey(1)
                if cv2.waitKey(1) & 0xff == ord('x'):
                    cv2.destroyWindow("Preview")
                    cv2.destroyWindow("Default")
                    break


class DetectionAndScoring(QRunnable):
    def closeEvent(self, event):
        super(QRunnable, self).closeEvent(event)
        self.ser.close()

    def __init__(self):
        global points, intersectp, ellipse_vertices, newpoints, intersectp_s, dart_point, TRIANGLE_DETECT_THRESH, useMovingAverage, score1, score2, scored_values, scored_mults, mults_of_round, values_of_round
        super().__init__()
        gui.create_gui()

    def run(self):
        global previous_img, difference, default_img, ACTIVE_PLAYER
        global points, intersectp, ellipse_vertices, newpoints, intersectp_s, dart_point, TRIANGLE_DETECT_THRESH, useMovingAverage, score1, score2, scored_values, scored_mults, mults_of_round, values_of_round
        global new_dart_point, update_dart_point
        while True:
            if STOP_DETECTION:
                break
            fpsReader = FPS()
            success, img = cap.read()
            if success:
                img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
                # cv2.imshow("Undist", img_undist)
                img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False)
                if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                    img_roi = cv2.resize(img_roi, resize_for_squish)
                    cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = gui.updateTrackBar()

                    detect_dart_circle_and_set_limits(img_roi=img_roi)

                    # get the difference image
                    difference = cv2.absdiff(img_roi, default_img)
                    # blur it for better edges
                    blur = cv2.GaussianBlur(difference, (5, 5), 0)
                    for i in range(10):
                        blur = cv2.GaussianBlur(blur, (9, 9), 1)
                    blur = cv2.bilateralFilter(blur, 9, 75, 75)
                    ret, thresh = cv2.threshold(blur, TRIANGLE_DETECT_THRESH, 255, 0)

                    if useMovingAverage:
                        images_for_rolling_average.append(thresh)
                        nr_imgs = 10
                        if len(images_for_rolling_average) > nr_imgs:
                            out = images_for_rolling_average[-1]
                            for j in range(nr_imgs - 1, 0, -1):
                                # out = cv2.add(images_for_rolling_average[j], out)
                                out = cv2.addWeighted(images_for_rolling_average[j], 0.5, out, 0.5, 1)
                            images_for_rolling_average = images_for_rolling_average[1:20]
                        else:
                            out = thresh
                    else:
                        out = thresh

                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    minArea = 800
                    for i in contours:
                        area = cv2.contourArea(i)
                        if area > minArea:
                            points_list = i.reshape(i.shape[0], i.shape[2])
                            triangle = cv2.minEnclosingTriangle(cv2.UMat(points_list.astype(np.float32)))
                            triangle_np_array = cv2.UMat.get(triangle[1])
                            if triangle_np_array is not None:
                                pt1, pt2, pt3 = triangle_np_array.astype(np.int32)  # TODO: ERROR
                            else:
                                pt1, pt2, pt3 = np.array([-1, -1]), np.array([-1, -1]), np.array([-1, -1])
                            # Display the triangles
                            cv2.line(thresh, pt1.ravel(), pt2.ravel(), (255, 0, 255), 2)
                            cv2.line(thresh, pt2.ravel(), pt3.ravel(), (255, 0, 255), 2)
                            cv2.line(thresh, pt3.ravel(), pt1.ravel(), (255, 0, 255), 2)

                            dart_point, rest_pts = dart_scorer_util.findTipOfDart(pt1, pt2, pt3)

                            # Display the Dart point
                            cv2.circle(thresh, dart_point, 4, (0, 0, 255), -1)
                            cv2.circle(img_roi, dart_point, 4, (0, 0, 255), -1)

                            radius, angle = dart_scorer_util.getRadiusAndAngle(center_ellipse[0], center_ellipse[1], dart_point[0], dart_point[1])
                            value, mult = dart_scorer_util.evaluateThrow(radius, angle)

                            bottom_point = dart_scorer_util.getBottomPoint(rest_pts[0], rest_pts[1], dart_point)
                            cv2.line(img_roi, dart_point, bottom_point, (0, 0, 255), 2)

                            k = -0.215  # scaling factor
                            vect = (dart_point - bottom_point)
                            new_dart_point = dart_point + k * vect
                            update_dart_point = True

                            cv2.circle(img_roi, new_dart_point.astype(np.int32), 4, (0, 255, 0), -1)
                            new_radius, new_angle = dart_scorer_util.getRadiusAndAngle(center_ellipse[0], center_ellipse[1], new_dart_point[0], new_dart_point[1])
                            new_val, new_mult = dart_scorer_util.evaluateThrow(new_radius, new_angle)

                            if len(scored_values) <= 20:
                                scored_values.append(new_val)
                                scored_mults.append(new_mult)
                            else:
                                final_val = mode(scored_values)
                                final_mult = mode(scored_mults)
                                values_of_round.append(final_val)
                                mults_of_round.append(final_mult)
                                default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                                print(f"Final val {final_val}")
                                print(f"Final mult {final_mult}")
                                if len(values_of_round) == 3:
                                    if ACTIVE_PLAYER == 1:
                                        dart_scorer_util.update_score(score1, values_of_round=values_of_round, mults_of_round=mults_of_round)
                                        ACTIVE_PLAYER = 2
                                    elif ACTIVE_PLAYER == 2:
                                        dart_scorer_util.update_score(score2, values_of_round=values_of_round, mults_of_round=mults_of_round)
                                        ACTIVE_PLAYER = 1
                                    values_of_round = []
                                    mults_of_round = []

                                scored_values = []
                                scored_mults = []

                    cv2.imshow("Threshold", out)

                    previous_img = img_roi
                    # TODO: Separate show image and processing image with cv2.copy
                    # cv2.ellipse(img_roi, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, (255, 0, 0))
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_1 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_2 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_3 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_4 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_5 / 100)), (255, 0, 255), 1)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_6 / 100)), (255, 0, 255), 1)

                    cv2.ellipse(img_roi, ellipse, (0, 255, 0), 2)
                    if dart_point is not None:
                        cv2.line(img_roi, center_ellipse, dart_point, (255, 0, 0), 2)

                    fps, img_roi = fpsReader.update(img_roi)
                    cv2.imshow("Dart Settings", utils.rez(img_roi, 1.5))
                else:
                    print("NO MARKERS FOUND!")
                    cv2.putText(img_undist, "NO MARKERS FOUND", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
                    cv2.imshow("Dart Settings", img_undist)
                    sleep(1)

                cv2.waitKey(1)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    cap.release()
                    exit()


class UIFunctions(QMainWindow):

    def undo_last_throw(self):
        global values_of_round, mults_of_round
        if len(values_of_round) > 0:
            values_of_round.pop()
            mults_of_round.pop()
            list(self.DartPositions.values())[-1].setText("")

    def delete_all_x_on_board(self):
        for lable in self.DartPositions.values():
            print("Test")
            print("Deleting: " + lable.text())
            lable.setText("")

    def place_x_on_board(self, pos_x, pos_y):
        global dart_id
        DartPositionId = dart_id
        dart_id = dart_id + 1
        self.DartPositions[DartPositionId] = DartPositionLabel(self.ui.dart_board_image)
        self.DartPositions[DartPositionId].addDartPosition(pos_x, pos_y)

    def set_default_image(self):
        pool = QThreadPool.globalInstance()
        default_img_setter = DefaultImageSetter()
        pool.start(default_img_setter)

    def start_detection_and_scoring(self):
        global STOP_DETECTION
        STOP_DETECTION = False
        pool = QThreadPool.globalInstance()
        detection_and_scoring = DetectionAndScoring()
        pool.start(detection_and_scoring)

    def stop_detection_and_scoring(self):
        global STOP_DETECTION
        STOP_DETECTION = True

    def update_labels(self):
        global values_of_round, mults_of_round, ACTIVE_PLAYER, new_dart_point, update_dart_point
        if update_dart_point:
            UIFunctions.place_x_on_board(self, new_dart_point[0], new_dart_point[1])
            update_dart_point = False
            print(f"New dart point !!!!!!!!!!!!!!!!!!!!!!!! {new_dart_point}")
        if ACTIVE_PLAYER == 1:
            self.ui.player1_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
            elif len(values_of_round) == 2:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
            elif len(values_of_round) == 3:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                UIFunctions.delete_all_x_on_board(self)
            else:
                self.ui.player1_1.setText("-")
                self.ui.player1_2.setText("-")
                self.ui.player1_3.setText("-")
        elif ACTIVE_PLAYER == 2:
            self.ui.player2_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
            elif len(values_of_round) == 2:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
            elif len(values_of_round) == 3:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                UIFunctions.delete_all_x_on_board(self)
            else:
                self.ui.player2_1.setText("-")
                self.ui.player2_2.setText("-")
                self.ui.player2_3.setText("-")


        self.ui.player1_overall.setText(str(score1.currentScore))
        self.ui.player2_overall.setText(str(score2.currentScore))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    label_update_timer = QtCore.QTimer()
    label_update_timer.timeout.connect(lambda: UIFunctions.update_labels(window))
    label_update_timer.start(10)  # every 10 milliseconds

    sys.exit(app.exec_())
