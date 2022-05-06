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

import DartScore
import math

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# #############  Config  ####################
saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True
webcam = True
rows = 6            # 17   6
columns = 9         # 28    9
squareSize = 30     # mm
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


def getRadiusAndAngle(centerX, centerY, pointX, pointY):
    """
    get the radius and the angle of the thrown point in relation to the board center
    """
    radius = -1.0   # indicates an error state
    angle = 0.0
    if (centerX >= 0) and (centerY >= 0) and (pointX >= 0) and (pointY >= 0):
        radius = math.sqrt((pointX-centerX)**2 + (pointY-centerY)**2)
        angle = math.asin((pointY-centerY) / radius)
        angle = angle * (180/math.pi)   # convert radiant to degrees
    return radius, angle


# dictionary that maps the angles to the corresponding fields on the dart board
# every field has a range of 18° and the first field is split in half,
# because the dart board does not start horizontally
listOfFields = {
        (0.0001, 9.000): 6,
        (9.0001, 27.000): 13,
        (27.0001, 45.000): 4,
        (45.0001, 63.000): 18,
        (63.0001, 81.000): 1,
        (81.0001, 99.000): 20,
        (99.0001, 117.000): 5,
        (117.0001, 135.000): 12,
        (135.0001, 153.000): 9,
        (153.0001, 171.000): 14,
        (171.0001, 189.000): 11,
        (189.0001, 207.000): 8,
        (207.0001, 225.000): 16,
        (225.0001, 243.000): 7,
        (243.0001, 261.000): 19,
        (261.0001, 279.000): 3,
        (279.0001, 297.000): 17,
        (297.0001, 315.000): 2,
        (315.0001, 333.000): 15,
        (333.0001, 351.000): 10,
        (351.0001, 360.000): 6,
    }

# radius limits for the different fields on the board
bullsLimit = 10.0
singleBullsLimit = 15.0   # example values
innerTripleLimit = 50.0
outerTripleLimit = 55.0
innerDoubleLimit = 95.0
outerBoardLimit = 100.0


def evaluateThrow(radius, angle):
    """
    evaluates the value and the multiplier of the field with given radius and angle
    """
    value = -1      # -1 is error state
    multiplier = 1

    if radius >= 0.0:
        if radius < outerBoardLimit:
            # evaluates the value of the field
            for limits in listOfFields:
                if limits[0] <= angle <= limits[1]:
                    value = listOfFields[limits]

            # evaluates the multiplier of the field
            if radius <= bullsLimit:
                return 50, 1    # Bull´s Eye!
            elif radius <= singleBullsLimit:
                return 25, 1    # Single Bull!
            elif radius <= innerTripleLimit:
                multiplier = 1
            elif radius <= outerTripleLimit:
                multiplier = 3
            elif radius <= innerDoubleLimit:
                multiplier = 1
            elif radius <= outerBoardLimit:
                multiplier = 2
        else:
            return 0, 1    # Dart is off the board, no points for player
    else:
        print("Radius is invalid! Point cannot be evaluated!")
        print("Please throw again!")

    return value, multiplier


(radius, angle) = getRadiusAndAngle(0, 0, 54, 10)
print(radius, angle)
print(evaluateThrow(radius, angle))


def create_gui():
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
    cv2.createTrackbar("Show Filters", "General Settings", 1, 1, empty)
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


# OpenCV Window GUI###############################
gui.create_gui()

# #################  Program Starting Screen  ########################
keyEvent = cv2.waitKey(0)   # next image
if keyEvent == ord('1'):    # calibrate and save
    saveParametersPickle = True
    loadSavedParameters = False
elif keyEvent == ord('2'):      # just calibrate
    saveParametersPickle = False
    loadSavedParameters = False
elif keyEvent == ord('3'):      # measure
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
    meanMTX, meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(cap=cap, rows=rows, columns=columns, squareSize=squareSize, runs=calibrationRuns,
                                                                                                    saveImages=False, webcam=webcam)
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

# Firs While Loop to set the default reference img
while True:
    succsess, img = cap.read()
    if succsess:
        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
        cv2.putText(img_undist, "Press q to take choose an image shown in 'Default' as default", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
        cv2.imshow("Preview",img_undist)
        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size,use_outer_corners=True)
        if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            default_img = img_roi
            print("Set default image")
            cv2.imshow("Default",default_img)
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

        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size,use_outer_corners=True)

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

                    #vx, vy, x, y = cv2.fitLine(cnt[1],cv2.DIST_L2,0,0.01,0.01)
                    #left_point = int((-x * vy / vx) + y)
                    difference = cv2.absdiff(img_roi, default_img)
                    blur = cv2.GaussianBlur(difference, (5, 5),0)
                    for i in range(radius_2):
                        blur = cv2.GaussianBlur(blur, (9, 9), 1)
                    blur = cv2.bilateralFilter(blur, 9, 75, 75)
                    ret, thresh = cv2.threshold(blur, radius_1, 255, 0)
                    gray = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
                    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    minArea = 100
                    for i in contours:
                        area = cv2.contourArea(i)
                        if area > minArea:
                            points_list = i.reshape(i.shape[0],i.shape[2])
                            triangle = cv2.minEnclosingTriangle(cv2.UMat(points_list.astype(np.float32)))
                            triangle_np_array = cv2.UMat.get(triangle[1])
                            pt1, pt2, pt3 = triangle_np_array.astype(np.int32)

                            # find tip of dart
                            dart_point = pt1
                            dist_1_2 = np.linalg.norm(pt1-pt2)
                            dist_1_3 = np.linalg.norm(pt1-pt3)
                            dist_2_3 = np.linalg.norm(pt2-pt3)
                            if dist_1_2 > dist_1_3 and dist_2_3 > dist_1_3:
                                dart_point = pt2
                            elif dist_1_3 > dist_1_2 and dist_2_3 > dist_1_2:
                                dart_point = pt3

                            cv2.circle(thresh, dart_point.ravel(), 16, (0,0,255), -1)
                            cv2.circle(img_roi, dart_point.ravel(), 16, (0,0,255), -1)


                            pt1_new = pt1.ravel()
                            cv2.line(thresh, pt1.ravel(), pt2.ravel(), (255, 0, 255), 2)
                            cv2.line(thresh, pt2.ravel(), pt3.ravel(), (255, 0, 255), 2)
                            cv2.line(thresh, pt3.ravel(), pt1.ravel(), (255, 0, 255), 2)

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
                    #TODO: Seperate show image and processing image with cv2.copy
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
        else:
            print("NO MARKERS FOUND!")
            cv2.putText(img_undist,"NO MARKERS FOUND",(300,300),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)
            cv2.imshow("Dart Settings", img_undist)

        cv2.waitKey(1)
