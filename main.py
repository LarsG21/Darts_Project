import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle
from FPS import FPS
import CalibrationWithUncertainty
import ContourUtils
from utils import rez
import gui
import utils
from CalibrationWithUncertainty import *
# import pytesseract
import queue
import dart_scorer_util

import DartScore
import math

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# #############  Config  ####################
saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True
webcam = True
rows = 6  # 17   6
columns = 9  # 28    9
squareSize = 30  # mm
calibrationRuns = 1

# #############  Config  #####################

points = []
intersectp = []
ellipse_vertices = []
newpoints = []
intersectp_s = []

# ####### Score - Test ###########
# score = DartScore.Score(501,True)
# points = score.calculatePoints(17,1,20,2,0,1)
# score.pointsScored(points[0], points[1])
# print(score.currentScore)

# points = score.calculatePoints(20,3,20,3,20,3)
# score.pointsScored(points[0], points[1])
# print(score.currentScore)

# points = score.calculatePoints(20,3,20,3,20,3)
# score.pointsScored(points[0], points[1])
# print(score.currentScore)

# points = score.calculatePoints(19,3,2,3,1,3)
# score.pointsScored(points[0], points[1])
# print(score.currentScore)

# points = score.calculatePoints(0,1,0,2,9,2)
# score.pointsScored(points[0], points[1])
# print(score.currentScore)
#############################################


# OpenCV Window GUI###############################
gui.create_gui()

# #################  Program Starting Screen  ########################
keyEvent = cv2.waitKey(0)  # next image
if keyEvent == ord('1'):  # calibrate and save
    saveParametersPickle = True
    loadSavedParameters = False
elif keyEvent == ord('2'):  # just calibrate
    saveParametersPickle = False
    loadSavedParameters = False
elif keyEvent == ord('3'):  # measure
    saveParametersPickle = False
    loadSavedParameters = True
elif keyEvent == ord('q'):
    exit()
else:
    cv2.waitKey(1)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not loadSavedParameters:
    meanMTX, meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(cap=cap, rows=rows,
                                                                                                    columns=columns,
                                                                                                    squareSize=squareSize,
                                                                                                    runs=calibrationRuns,
                                                                                                    saveImages=False,
                                                                                                    webcam=webcam)
if saveParametersPickle:
    pickle_out_MTX = open("PickleFiles/mtx.pickle", "wb")
    pickle.dump(meanMTX, pickle_out_MTX)
    pickle_out_MTX.close()
    pickle_out_DIST = open("PickleFiles/dist.pickle", "wb")
    pickle.dump(meanDIST, pickle_out_DIST)
    pickle_out_DIST.close()
    pickle_out_MTX_Un = open("PickleFiles/uncertaintyMtx.pickle", "wb")
    pickle.dump(uncertaintyMTX, pickle_out_MTX_Un)
    pickle_out_MTX_Un.close()
    pickle_out_DIST_Un = open("PickleFiles/uncertaintyDist.pickle", "wb")
    pickle.dump(uncertaintyDIST, pickle_out_DIST_Un)
    pickle_out_DIST_Un.close()
    print("Parameters Saved")

if loadSavedParameters:
    pickle_in_MTX = open("PickleFiles/mtx.pickle", "rb")
    meanMTX = pickle.load(pickle_in_MTX)
    print(meanMTX)
    pickle_in_DIST = open("PickleFiles/dist.pickle", "rb")
    meanDIST = pickle.load(pickle_in_DIST)
    print(meanDIST)
    print("Parameters Loaded")

#####################################################################################


target_ROI_size = (600, 600)
previos_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)

default_img = np.zeros(target_ROI_size).astype(np.uint8)

cv2.destroyWindow("Object measurement")

# Firs While Loop to set the default reference img
while True:
    succsess, img = cap.read()
    if succsess:
        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
        cv2.putText(img_undist, "Press q to take choose an image shown in 'Default' as default", (5, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
        cv2.imshow("Preview", img_undist)
        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=True)
        if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            default_img = img_roi
            print("Set default image")
            cv2.imshow("Default", default_img)
            cv2.waitKey(1)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyWindow("Preview")
            cv2.destroyWindow("Default")
            break

while True:
    fpsReader = FPS()
    succsess, img = cap.read()
    if succsess:
        print(img.shape)
        # cv2.imshow("Originaimg",img)
        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
        cv2.imshow("Undist", img_undist)

        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=True)

        if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:

            cannyLow, cannyHigh, noGauss, minArea, errosions, dialations, epsilon, showFilters, automaticMode, textSize = gui.updateTrackBar()

            imgContours, contours, imgCanny = ContourUtils.get_contours(img=img_roi, cThr=(cannyLow, cannyHigh),
                                                                        gaussFilters=noGauss, minArea=minArea,
                                                                        epsilon=epsilon, draw=False,
                                                                        errsoions=errosions, dialations=dialations,
                                                                        showFilters=showFilters)  # gets Contours from Image

            cv2.imshow("Contours", imgContours)

            for cnt in contours:
                if 200000 / 4 < cnt[1] < 1000000 / 4:
                    radius_1, radius_2, radius_3, x_offset, y_offset = gui.update_dart_trackbars()

                    # vx, vy, x, y = cv2.fitLine(cnt[1],cv2.DIST_L2,0,0.01,0.01)
                    # left_point = int((-x * vy / vx) + y)
                    difference = cv2.absdiff(img_roi, default_img)
                    blur = cv2.GaussianBlur(difference, (5, 5), 0)
                    for i in range(radius_2):
                        blur = cv2.GaussianBlur(blur, (9, 9), 1)
                    blur = cv2.bilateralFilter(blur, 9, 75, 75)
                    ret, thresh = cv2.threshold(blur, radius_1, 255, 0)
                    imgs = []
                    imgs.insert(0, thresh)
                    nr_imgs = 20
                    if len(imgs) > nr_imgs:
                        out = imgs[0]
                        for j in range(1, nr_imgs):
                            out = cv2.add(img[j], out)
                        imgs.pop()
                    else:
                        out = thresh
                    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    minArea = 100
                    for i in contours:
                        area = cv2.contourArea(i)
                        if area > minArea:
                            points_list = i.reshape(i.shape[0], i.shape[2])
                            triangle = cv2.minEnclosingTriangle(cv2.UMat(points_list.astype(np.float32)))
                            triangle_np_array = cv2.UMat.get(triangle[1])
                            pt1, pt2, pt3 = triangle_np_array.astype(np.int32)

                            # find tip of dart
                            dart_point = pt1
                            dist_1_2 = np.linalg.norm(pt1 - pt2)
                            dist_1_3 = np.linalg.norm(pt1 - pt3)
                            dist_2_3 = np.linalg.norm(pt2 - pt3)
                            if dist_1_2 > dist_1_3 and dist_2_3 > dist_1_3:
                                dart_point = pt2
                            elif dist_1_3 > dist_1_2 and dist_2_3 > dist_1_2:
                                dart_point = pt3

                            cv2.circle(thresh, dart_point.ravel(), 16, (0, 0, 255), -1)
                            cv2.circle(img_roi, dart_point.ravel(), 16, (0, 0, 255), -1)

                            pt1_new = pt1.ravel()
                            cv2.line(thresh, pt1.ravel(), pt2.ravel(), (255, 0, 255), 2)
                            cv2.line(thresh, pt2.ravel(), pt3.ravel(), (255, 0, 255), 2)
                            cv2.line(thresh, pt3.ravel(), pt1.ravel(), (255, 0, 255), 2)

                    cv2.imshow("Threshold", out)
                    x = 3000
                    # if cv2.countNonZero(thresh) > x and cv2.countNonZero(thresh) < 15000:  ## threshold important -> make accessible

                    ellipse = cv2.fitEllipse(cnt[4])
                    cv2.ellipse(img_roi, ellipse, (0, 255, 0), 5)

                    x, y = ellipse[0]
                    a, b = ellipse[1]
                    angle = ellipse[2]

                    center_ellipse = (int(x + x_offset / 10), int(y + y_offset / 10))

                    a = a / 2
                    b = b / 2

                    previos_img = img_roi
                    # TODO: Seperate show image and processing image with cv2.copy
                    cv2.ellipse(img_roi, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, (255, 0, 0))
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_1 / 100)), (255, 0, 255), 2)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_2 / 100)), (255, 0, 255), 2)
                    cv2.circle(img_roi, center_ellipse, int(a * (radius_3 / 100)), (255, 0, 255), 2)

                    # cv2.circle(image_proc_img, (int(x), int(y-b/2)), 3, cv.CV_RGB(0, 255, 0), 2, 8)

                    # vertex calculation
                    xb = b * math.cos(angle)
                    yb = b * math.sin(angle)

                    xa = a * math.sin(angle)
                    ya = a * math.cos(angle)

                    rect = cv2.minAreaRect(cnt[4])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # cv2.drawContours(img_undist, [box], 0, (0, 0, 255), 2)
            fps, img_roi = fpsReader.update(img_roi)
            cv2.imshow("Dart Settings", rez(img_roi, 2))
            # cv2.imshow("Diff", difference)
        else:
            print("NO MARKERS FOUND!")
            cv2.putText(img_undist, "NO MARKERS FOUND", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
            cv2.imshow("Dart Settings", img_undist)

        cv2.waitKey(1)
