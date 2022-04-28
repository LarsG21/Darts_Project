import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

from FPS import FPS

import CalibrationWithUncertainty
import ContourUtils
import gui
import utils
from utils import rez
from CalibrationWithUncertainty import *
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

################################Config####################
saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True
webcam = True
rows = 17  # 17   6
columns = 28  # 28    9
squareSize = 10  # mm
calibrationRuns = 1

################################Config####################

points = []
intersectp = []
ellipse_vertices = []
newpoints = []
intersectp_s = []

# OpenCV Window GUI###############################
mainImage = cv2.imread("Recources/Main Frame.PNG")
root_wind = "Object measurement"
cv2.namedWindow(root_wind)
cv2.imshow(root_wind, mainImage)


def empty(a):
    pass


slider = "Edge Detection Settings"
filters = "General Settings"
dart_settings = "Dart Settings"
cv2.namedWindow(filters)
cv2.namedWindow(slider)
cv2.namedWindow(dart_settings)

cv2.resizeWindow("General Settings", 400, 100)
cv2.resizeWindow("Edge Detection Settings", 640, 240)
cv2.resizeWindow("Dart Settings", 640, 240)

cv2.createTrackbar("Edge Thresh Low", "Edge Detection Settings", 80, 255, empty)
cv2.createTrackbar("Edge Thresh High", "Edge Detection Settings", 160, 255, empty)
cv2.createTrackbar("Gaussian's", "Edge Detection Settings", 2, 20, empty)
cv2.createTrackbar("Dilations", "Edge Detection Settings", 1, 10, empty)
cv2.createTrackbar("Erosions", "Edge Detection Settings", 1, 10, empty)
cv2.createTrackbar("minArea", "Edge Detection Settings", 800, 500000, empty)
cv2.createTrackbar("Epsilon", "Edge Detection Settings", 5, 40, empty)
cv2.createTrackbar("Show Filters", "General Settings", 0, 1, empty)
cv2.createTrackbar("Automatic", "General Settings", 0, 1, empty)
cv2.createTrackbar("TextSize", "General Settings", 100, 400, empty)

cv2.createTrackbar("Circle1", "Dart Settings", 100, 100, empty)
cv2.createTrackbar("Circle2", "Dart Settings", 100, 100, empty)
cv2.createTrackbar("Circle3", "Dart Settings", 100, 100, empty)
cv2.createTrackbar("Circle3", "Dart Settings", 100, 100, empty)

cv2.createTrackbar("X_Offset", "Dart Settings", 0, 100, empty)
cv2.setTrackbarMin("X_Offset", "Dart Settings", -100)
cv2.createTrackbar("Y_Offset", "Dart Settings", 0, 100, empty)
cv2.setTrackbarMin("Y_Offset", "Dart Settings", -100)

#######################################################################


#################################Program Starting Screen#####################################
keyEvent = cv2.waitKey(0)  # next imageqq
if keyEvent == ord('1'):  # calibrate and save
    saveParametersPickle = True
    loadSavedParameters = False
elif keyEvent == ord('2'):  # just calibrate
    saveParametersPickle = False
    loadSavedParameters = False
elif keyEvent == ord('3'):  # masure
    saveParametersPickle = False
    loadSavedParameters = True
elif keyEvent == ord('q'):
    exit()
else:
    cv2.waitKey(1)

cap = cv2.VideoCapture(2)
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

########################################################################################################

target_ROI_size = (600, 600)
previos_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)

default_img = np.zeros(target_ROI_size).astype(np.uint8)

while True:
    succsess, img = cap.read()
    if succsess:
        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
        cv2.imshow("Preview",img_undist)
        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size)
        if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            print("ROI_DETECTED")
            cv2.imshow("ROI",img_roi)
            cv2.waitKey(1)
            default_img = img_roi
            print("Set default image")
            cv2.imshow("Default",default_img)
            cv2.waitKey(1000)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyWindow("Preview")
            cv2.destroyWindow("Default")
            cv2.destroyWindow("ROI")
            break

while True:
    fpsReader = FPS()
    succsess, img = cap.read()
    if succsess:
        print(img.shape)
        # cv2.imshow("Originaimg",img)
        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
        cv2.imshow("Undist", img_undist)

        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size)

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

                    difference = cv2.absdiff(img_roi, default_img)
                    blur = cv2.GaussianBlur(difference, (5, 5), 0)
                    blur = cv2.bilateralFilter(blur, 9, 75, 75)
                    ret, thresh = cv2.threshold(blur, 60, 255, 0)
                    cv2.imshow("Threshold",thresh)
                    x = 3000
                    #if cv2.countNonZero(thresh) > x and cv2.countNonZero(thresh) < 15000:  ## threshold important -> make accessible

                    ellipse = cv2.fitEllipse(cnt[4])
                    cv2.ellipse(img_roi, ellipse, (0, 255, 0), 5)

                    x, y = ellipse[0]
                    a, b = ellipse[1]
                    angle = ellipse[2]

                    center_ellipse = (int(x + x_offset / 10), int(y + y_offset / 10))

                    a = a / 2
                    b = b / 2


                    previos_img = img_roi

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
            cv2.imshow("Diff", difference)
        cv2.waitKey(1)
