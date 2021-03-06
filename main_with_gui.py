import pickle
import sys
from statistics import mode
from time import sleep

import cv2
import numpy as np
from PySide2 import QtCore
from PySide2.QtCore import QThreadPool, QRunnable
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QMainWindow, QMessageBox

import CalibrationWithUncertainty
import ContourUtils
from Dart_Scoring import dart_scorer_util, DartScore
import gui
import utils
from FPS import FPS
from QT_GUI_Elements.qt_ui_classes import DartPositionLabel
from QT_GUI_Elements.ui_dart_main_gui import Ui_DartScorer

# Globals
points = []
intersectp = []
ellipse_vertices = []
newpoints = []
intersectp_s = []
dart_point = None
ACTIVE_PLAYER = 1
UNDO_LAST_FLAG = False

# #############  Config  ####################

loadSavedParameters = True
rows = 6  # 17   6
columns = 9  # 28    9
squareSize = 30  # mm
calibrationRuns = 1
CAMERA_NUMBER = 1  # 0,1 is built-in, 2 is external webcam
TRIANGLE_DETECT_THRESH = 11
minArea = 800
maxArea = 4000
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
resize_for_squish = (535, 600)

Scaling_factor_for_x_placing_in_gui = (501/resize_for_squish[0], 501/resize_for_squish[1])

previous_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)
img_undist = np.zeros(target_ROI_size).astype(np.uint8)

default_img = np.zeros(target_ROI_size).astype(np.uint8)


def detect_dart_circle_and_set_limits(img_roi):
    cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = gui.updateTrackBar()
    global contours, radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, cnt, ellipse, x, y, a, b, angle, center_ellipse, x_offset_current,\
        y_offset_current, TRIANGLE_DETECT_THRESH
    imgContours, contours, imgCanny = ContourUtils.get_contours(img=img_roi, cThr=(cannyLow, cannyHigh),
                                                                gaussFilters=noGauss, minArea=minArea,
                                                                epsilon=epsilon, draw=False,
                                                                erosions=erosions, dilations=dilations,
                                                                showFilters=showFilters)
    TRIANGLE_DETECT_THRESH, minArea, maxArea, radius_4, radius_5, radius_6, x_offset, y_offset = gui.update_dart_trackbars()
    radius_1, radius_2, radius_3 = 5, 10, 58
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
        self.setWindowIcon(QIcon('icons/dart_icon.ico'))
        self.setWindowTitle("Dart Master")


        # Buttons
        self.ui.set_default_img_button.clicked.connect(lambda: UIFunctions.set_default_image(self))
        self.ui.start_measuring_button.clicked.connect(lambda: UIFunctions.start_detection_and_scoring(self))
        self.ui.stop_measuring_button.clicked.connect(lambda: UIFunctions.stop_detection_and_scoring(self))
        self.ui.undo_last_throw_button.clicked.connect(lambda: UIFunctions.undo_last_throw(self))
        self.ui.initial_score_comboBox.currentIndexChanged.connect(lambda: UIFunctions.update_game_settings(self))
        self.ui.detection_sensitivity_slider.valueChanged.connect(lambda: UIFunctions.update_detection_sensitivity(self))
        self.ui.continue_button.clicked.connect(lambda: UIFunctions.start_detection_and_scoring(self))
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

    def warning(self, message="Default"):
        QMessageBox.about(self, "Congratulations !", message)


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
        global points, intersectp, ellipse_vertices, newpoints, intersectp_s, dart_point, TRIANGLE_DETECT_THRESH, \
            useMovingAverage, score1, score2, scored_values, scored_mults, mults_of_round, values_of_round, img_undist
        super().__init__()
        gui.create_gui()
        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
        cv2.destroyWindow("Object measurement")


    def run(self):
        global previous_img, difference, default_img, ACTIVE_PLAYER, UNDO_LAST_FLAG
        global points, intersectp, ellipse_vertices, newpoints, intersectp_s, dart_point, TRIANGLE_DETECT_THRESH, useMovingAverage, score1, score2, scored_values, scored_mults, mults_of_round, values_of_round
        global new_dart_point, update_dart_point, minArea
        while True:
            if STOP_DETECTION:
                break
            fpsReader = FPS()
            success, img = cap.read()
            if success:
                img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
                # cv2.imshow("Undist", img_undist)
                img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False, hold_position=True)
                if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
                    img_roi = cv2.resize(img_roi, resize_for_squish)
                    # resize img by a factor of 2
                    img_show = cv2.resize(img_roi, dsize=(400, 400))
                    cv2.imshow("Live", img_show)

                    cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = gui.updateTrackBar()

                    detect_dart_circle_and_set_limits(img_roi=img_roi)

                    # get the difference image
                    if np.all(default_img==0): #TODO: Bad fix but works
                        default_img = img_roi.copy()
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

                    # minArea = minArea*100
                    # maxArea = minArea*100
                    minArea = 600
                    for i in contours:
                        area = cv2.contourArea(i)
                        if minArea < area:
                            points_list = i.reshape(i.shape[0], i.shape[2])
                            triangle = cv2.minEnclosingTriangle(cv2.UMat(points_list.astype(np.float32)))
                            triangle_np_array = cv2.UMat.get(triangle[1])
                            if triangle_np_array is not None:
                                pt1, pt2, pt3 = triangle_np_array.astype(np.int32)
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

                            cv2.circle(img_roi, new_dart_point.astype(np.int32), 4, (0, 255, 0), -1)
                            new_radius, new_angle = dart_scorer_util.getRadiusAndAngle(center_ellipse[0], center_ellipse[1], new_dart_point[0], new_dart_point[1])
                            new_val, new_mult = dart_scorer_util.evaluateThrow(new_radius, new_angle)

                            if len(scored_values) <= 20:    # Wait for 20 Results
                                scored_values.append(new_val)
                                scored_mults.append(new_mult)
                            else:
                                if UNDO_LAST_FLAG:
                                    window.ui.press_enter_label.setText("")
                                update_dart_point = True
                                final_val = mode(scored_values) # Take the most frequent result and use that as the final result
                                final_mult = mode(scored_mults)
                                values_of_round.append(final_val)
                                mults_of_round.append(final_mult)
                                default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                                # print(f"Final val {final_val}")
                                # print(f"Final mult {final_mult}")
                                # Continue if enter is pressed
                                if len(values_of_round) == 3:
                                    UNDO_LAST_FLAG = False
                                    # if cv2.waitKey(0) & 0xFF == ord('\r'):
                                    # if window.ui.continue_button.isChecked():
                                    UIFunctions.stop_detection_and_scoring(window)
                                    success, img = cap.read()   #Reset the default image after every dart
                                    if success:
                                        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
                                    default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                                    if not UNDO_LAST_FLAG:
                                        if not STOP_DETECTION:
                                            window.ui.press_enter_label.setText("")
                                        if ACTIVE_PLAYER == 1:
                                            dart_scorer_util.update_score(score1, values_of_round=values_of_round, mults_of_round=mults_of_round)
                                            ACTIVE_PLAYER = 2
                                        elif ACTIVE_PLAYER == 2:
                                            dart_scorer_util.update_score(score2, values_of_round=values_of_round, mults_of_round=mults_of_round)
                                            ACTIVE_PLAYER = 1
                                        values_of_round = []
                                        mults_of_round = []
                                    else:
                                        UNDO_LAST_FLAG = False
                                    # else:
                                    #     pass


                                            # Reset the image
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
                    # if dart_point is not None:
                    #     cv2.line(img_roi, center_ellipse, dart_point, (255, 0, 0), 2)

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
            # UIFunctions.update_labels(window)


class UIFunctions(QMainWindow):

    def update_detection_sensitivity(self):
        global TRIANGLE_DETECT_THRESH
        TRIANGLE_DETECT_THRESH = self.ui.detection_sensitivity_slider.value()
        self.ui.current_detection_sensitivity_lable.setText(f"{TRIANGLE_DETECT_THRESH}")

    def undo_last_throw(self):
        global values_of_round, mults_of_round, UNDO_LAST_FLAG
        UNDO_LAST_FLAG = True
        self.ui.press_enter_label.setText("    Please throw again")
        if len(values_of_round) > 0:
            val = values_of_round.pop()
            mult = mults_of_round.pop()
            if ACTIVE_PLAYER == 1:
                self.ui.player1_sum_round.setText(str(int(self.ui.player1_sum_round.text()) - val * mult))
                if values_of_round == 1:
                    self.ui.player1_1_label.setText("")
                elif values_of_round == 2:
                    self.ui.player1_2_label.setText("")
                elif values_of_round == 3:
                    self.ui.player1_3_label.setText("")
            elif ACTIVE_PLAYER == 2:
                self.ui.player2_sum_round.setText(str(int(self.ui.player2_sum_round.text()) - val * mult))
                if values_of_round == 1:
                    self.ui.player2_1_label.setText("")
                elif values_of_round == 2:
                    self.ui.player2_2_label.setText("")
                elif values_of_round == 3:
                    self.ui.player2_3_label.setText("")

            list(self.DartPositions.values())[-1].setText("")


    def delete_all_x_on_board(self):
        print("LEN:", len(self.DartPositions.values()))
        for lable in self.DartPositions.values():
            lable.setText("")
        self.DartPositions = {}

    def place_x_on_board(self, pos_x, pos_y):
        global dart_id
        DartPositionId = dart_id
        dart_id = dart_id + 1
        self.DartPositions[DartPositionId] = DartPositionLabel(self.ui.dart_board_image)
        # print(f"Placing: {str(int(pos_x*Scaling_factor_for_x_placing_in_gui[0])), str(int(pos_y*Scaling_factor_for_x_placing_in_gui[1]))}")
        self.DartPositions[DartPositionId].addDartPosition(int(pos_x*Scaling_factor_for_x_placing_in_gui[0]),
                                                           int(pos_y*Scaling_factor_for_x_placing_in_gui[1]))

    def set_default_image(self):
        pool = QThreadPool.globalInstance()
        default_img_setter = DefaultImageSetter()
        pool.start(default_img_setter)

    def start_detection_and_scoring(self):
        global STOP_DETECTION, default_img, img_undist
        STOP_DETECTION = False
        # change color of stop_measuring_button to transparent
        self.ui.stop_measuring_button.setStyleSheet("background-color: transparent")
        # change color of start_measuring_button to green
        self.ui.start_measuring_button.setStyleSheet("background-color: green")
        window.ui.press_enter_label.setText("")
        pool = QThreadPool.globalInstance()
        default_img = utils.reset_default_image(img_undist, target_ROI_size, resize_for_squish)
        detection_and_scoring = DetectionAndScoring()
        UIFunctions.delete_all_x_on_board(window)################################
        pool.start(detection_and_scoring)

    def stop_detection_and_scoring(self):
        global STOP_DETECTION
        # change color of stop_measuring_button to red
        self.ui.stop_measuring_button.setStyleSheet("background-color: red")
        self.ui.start_measuring_button.setStyleSheet("background-color: transparent")
        window.ui.press_enter_label.setText("    1. Remove all Darts\n    2. Press Continue to start next round")
        STOP_DETECTION = True

    def update_game_settings(self):
        score = int(self.ui.initial_score_comboBox.currentText())
        score1.setNominalScore(score)
        score2.setNominalScore(score)


    def update_labels(self):
        global values_of_round, mults_of_round, ACTIVE_PLAYER, new_dart_point, update_dart_point
        if update_dart_point and new_dart_point is not None:
            # print(f"Updating dart point{new_dart_point[0], new_dart_point[1]}")
            UIFunctions.place_x_on_board(self, new_dart_point[0], new_dart_point[1])
            update_dart_point = False
        if ACTIVE_PLAYER == 1:
            self.ui.player_frame.setStyleSheet("background-color: #3a3a3a;")
            self.ui.player_frame_2.setStyleSheet("background-color: rgb(35, 35, 35);")
            # self.ui.player1_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0]*mults_of_round[0]))
            elif len(values_of_round) == 2:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0]*mults_of_round[0] + values_of_round[1]*mults_of_round[1]))
            elif len(values_of_round) == 3:
                self.ui.player1_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player1_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player1_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                self.ui.player1_sum_round.setText(str(values_of_round[0]*mults_of_round[0] + values_of_round[1]*mults_of_round[1] + values_of_round[2]*mults_of_round[2]))
            else:
                self.ui.player1_1.setText("-")
                self.ui.player1_2.setText("-")
                self.ui.player1_3.setText("-")
                self.ui.player2_sum_round.setText("")
        elif ACTIVE_PLAYER == 2:
            self.ui.player_frame_2.setStyleSheet("background-color: #3a3a3a;")
            self.ui.player_frame.setStyleSheet("background-color: rgb(35, 35, 35);")
            self.ui.player2_sum_round.setText(str(sum(values_of_round)))
            if len(values_of_round) == 1:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0]*mults_of_round[0]))
            elif len(values_of_round) == 2:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0]*mults_of_round[0] + values_of_round[1]*mults_of_round[1]))
            elif len(values_of_round) == 3:
                self.ui.player2_1.setText(f"{values_of_round[0] * mults_of_round[0]}")
                self.ui.player2_2.setText(f"{values_of_round[1] * mults_of_round[1]}")
                self.ui.player2_3.setText(f"{values_of_round[2] * mults_of_round[2]}")
                self.ui.player2_sum_round.setText(str(values_of_round[0]*mults_of_round[0] + values_of_round[1]*mults_of_round[1] + values_of_round[2]*mults_of_round[2]))
            else:
                self.ui.player2_1.setText("-")
                self.ui.player2_2.setText("-")
                self.ui.player2_3.setText("-")
                self.ui.player2_sum_round.setText("")

        # if one of the players has won the game, show the winner
        if score1.currentScore == 0:
            window.warning("Player 1 has won the game!")
        elif score2.currentScore == 0:
            window.warning("Player 2 has won the game!")

        self.ui.player1_overall.setText(str(score1.currentScore))
        self.ui.player2_overall.setText(str(score2.currentScore))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    label_update_timer = QtCore.QTimer()
    label_update_timer.timeout.connect(lambda: UIFunctions.update_labels(window))
    label_update_timer.start(10)  # every 10 milliseconds


    sys.exit(app.exec_())
