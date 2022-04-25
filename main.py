import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle

import CalibrationWithUncertainty
import ContourUtils
import gui
import utils
from CalibrationWithUncertainty import *
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

################################Config####################
saveImages = False
undistiortTestAfterCalib = False
saveParametersPickle = False
loadSavedParameters = True
webcam = True
rows = 6            #17   6
columns = 9         #28    9
squareSize = 30 #mm
calibrationRuns = 1


################################Config####################

points = []
intersectp = []
ellipse_vertices = []
newpoints = []
intersectp_s = []


#OpenCV Window GUI###############################
mainImage = cv2.imread("Recources/Main Frame.PNG")
root_wind = "Object measurement"
cv2.namedWindow(root_wind)
cv2.imshow(root_wind,mainImage)

def empty(a):
    pass
slider = "Edge Detection Settings"
filters = "General Settings"
dart_settings = "Dart Settings"
cv2.namedWindow(filters)
cv2.namedWindow(slider)
cv2.namedWindow(dart_settings)

cv2.resizeWindow("General Settings",400,100)
cv2.resizeWindow("Edge Detection Settings", 640, 240)
cv2.resizeWindow("Dart Settings", 640, 240)

cv2.createTrackbar("Edge Thresh Low","Edge Detection Settings", 80, 255, empty)
cv2.createTrackbar("Edge Thresh High","Edge Detection Settings", 160, 255, empty)
cv2.createTrackbar("Gaussian's","Edge Detection Settings", 2, 20, empty)
cv2.createTrackbar("Dilations","Edge Detection Settings", 1, 10, empty)
cv2.createTrackbar("Erosions","Edge Detection Settings", 1, 10, empty)
cv2.createTrackbar("minArea","Edge Detection Settings", 800, 500000, empty)
cv2.createTrackbar("Epsilon","Edge Detection Settings", 5, 40, empty)
cv2.createTrackbar("Show Filters","General Settings", 1, 1, empty)
cv2.createTrackbar("Automatic","General Settings",0,1,empty)
cv2.createTrackbar("TextSize","General Settings",100,400,empty)

cv2.createTrackbar("Circle1","Dart Settings",100,100,empty)
cv2.createTrackbar("Circle2","Dart Settings",100,100,empty)
cv2.createTrackbar("Circle3","Dart Settings",100,100,empty)
cv2.createTrackbar("Circle3","Dart Settings",100,100,empty)

cv2.createTrackbar("X_Offset","Dart Settings",0,100,empty)
cv2.setTrackbarMin("X_Offset", "Dart Settings", -100)
cv2.createTrackbar("Y_Offset","Dart Settings",0,100,empty)
cv2.setTrackbarMin("Y_Offset", "Dart Settings", -100)



#######################################################################


#################################Program Starting Screen#####################################
keyEvent = cv2.waitKey(0) #next imageqq
if keyEvent == ord('1'):            #calibrate and save
    saveParametersPickle = True
    loadSavedParameters = False
elif keyEvent == ord('2'):          #just calibrate
    saveParametersPickle = False
    loadSavedParameters = False
elif keyEvent == ord('3'):      #masure
    saveParametersPickle = False
    loadSavedParameters = True
elif keyEvent == ord('q'):
    exit()
else:
    cv2.waitKey(1)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


if not loadSavedParameters:
    meanMTX, meanDIST, uncertaintyMTX, uncertaintyDIST = CalibrationWithUncertainty.calibrateCamera(cap=cap, rows=rows, columns=columns, squareSize=squareSize, runs=calibrationRuns,
                                                                                                    saveImages=False, webcam=webcam)
if saveParametersPickle:
    pickle_out_MTX = open("PickleFiles/mtx.pickle","wb")
    pickle.dump(meanMTX,pickle_out_MTX)
    pickle_out_MTX.close()
    pickle_out_DIST = open("PickleFiles/dist.pickle","wb")
    pickle.dump(meanDIST,pickle_out_DIST)
    pickle_out_DIST.close()
    pickle_out_MTX_Un = open("PickleFiles/uncertaintyMtx.pickle", "wb")
    pickle.dump(uncertaintyMTX, pickle_out_MTX_Un)
    pickle_out_MTX_Un.close()
    pickle_out_DIST_Un = open("PickleFiles/uncertaintyDist.pickle", "wb")
    pickle.dump(uncertaintyDIST, pickle_out_DIST_Un)
    pickle_out_DIST_Un.close()
    print("Parameters Saved")

if loadSavedParameters:
    pickle_in_MTX = open("PickleFiles/mtx.pickle","rb")
    meanMTX = pickle.load(pickle_in_MTX)
    print(meanMTX)
    pickle_in_DIST = open("PickleFiles/dist.pickle", "rb")
    meanDIST = pickle.load(pickle_in_DIST)
    print(meanDIST)
    print("Parameters Loaded")


########################################################################################################


while True:

    succsess, img = cap.read()
    # print(img.shape)
    if succsess:
        # cv2.imshow("Originaimg",img)
        img_undist = utils.undistortFunction(img,meanMTX,meanDIST)
        cv2.imshow("Undist",img_undist)

        img_undist = ContourUtils.extract_roi_from_4_aruco_markers(img_undist,(600,600))

        if img_undist is not None and img_undist.shape[1] > 0 and img_undist.shape[0] > 0:

            cannyLow, cannyHigh, noGauss, minArea, errosions, dialations, epsilon, showFilters, automaticMode, textSize = gui.updateTrackBar()

            imgContours, contours, imgCanny = ContourUtils.get_contours(img=img_undist, cThr=(cannyLow, cannyHigh), gaussFilters=noGauss, minArea=minArea, epsilon=epsilon, draw=False,
                                                            errsoions=errosions,dialations=dialations,showFilters=showFilters)  # gets Contours from Image

            cv2.imshow("Contours", imgContours)

            for cnt in contours:
                if 200000/4 < cnt[1] < 1000000/4:
                    radius_1,radius_2,radius_3,x_offset,y_offset = gui.update_dart_trackbars()

                    ellipse = cv2.fitEllipse(cnt[4])
                    cv2.ellipse(img_undist, ellipse, (0, 255, 0), 5)

                    x, y = ellipse[0]
                    a, b = ellipse[1]
                    angle = ellipse[2]

                    center_ellipse = (int(x+x_offset/10), int(y+y_offset/10))

                    a = a/2
                    b = b/2

                    cv2.ellipse(img_undist, (int(x), int(y)), (int(a), int(b)), int(angle), 0.0, 360.0, (255, 0, 0))


                    cv2.circle(img_undist,center_ellipse,int(a*(radius_1/100)),(255,0,255),2)
                    cv2.circle(img_undist, center_ellipse, int(a * (radius_2 / 100)), (255, 0, 255), 2)
                    cv2.circle(img_undist, center_ellipse, int(a * (radius_3 / 100)), (255, 0, 255), 2)

                    #cv2.circle(image_proc_img, (int(x), int(y-b/2)), 3, cv.CV_RGB(0, 255, 0), 2, 8)

                    # vertex calculation
                    xb = b * math.cos(angle)
                    yb = b * math.sin(angle)

                    xa = a * math.sin(angle)
                    ya = a * math.cos(angle)

                    rect = cv2.minAreaRect(cnt[4])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # cv2.drawContours(img_undist, [box], 0, (0, 0, 255), 2)


            # circle_radius = a
            #
            # anglezone1 = (angle - 5, angle + 5)
            # anglezone2 = (angle - 100, angle - 80)
            #
            # # transform ellipse to a perfect circle?
            # height, width = img_undist.shape[:2]
            #
            # angle = (angle) * math.pi / 180
            #
            # # build transformation matrix http://math.stackexchange.com/questions/619037/circle-affine-transformation
            # R1 = np.array([[math.cos(angle), math.sin(angle), 0], [-math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            # R2 = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
            #
            # T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
            # T2 = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
            #
            # D = np.array([[1, 0, 0], [0, a / b, 0], [0, 0, 1]])
            #
            # M = T2.dot(R2.dot(D.dot(R1.dot(T1))))
            #
            # M_inv = np.linalg.inv(M)
            #
            # # fit line to find intersec point for dartboard center point
            # # change houghline parameter of angle
            # lines = cv2.HoughLines(imgCanny, 1, np.pi / 70, 100, 100)
            #
            # p = []
            # lines_seg = []
            # counter = 0
            # for rho, theta in lines[0]:
            #     # split between horizontal and vertical lines (take only lines in certain range)
            #     if theta > np.pi / 180 * anglezone1[0] and theta < np.pi / 180 * anglezone1[1]:
            #
            #         a = np.cos(theta)
            #         b = np.sin(theta)
            #         x0 = a * rho
            #         y0 = b * rho
            #         x1 = int(x0 + 3000 * (-b))
            #         y1 = int(y0 + 3000 * (a))
            #         x2 = int(x0 - 3000 * (-b))
            #         y2 = int(y0 - 3000 * (a))
            #
            #         for rho1, theta1 in lines[0]:
            #
            #             if theta1 > np.pi / 180 * anglezone2[0] and theta1 < np.pi / 180 * anglezone2[1]:
            #
            #                 a = np.cos(theta1)
            #                 b = np.sin(theta1)
            #                 x0 = a * rho1
            #                 y0 = b * rho1
            #                 x3 = int(x0 + 3000 * (-b))
            #                 y3 = int(y0 + 3000 * (a))
            #                 x4 = int(x0 - 3000 * (-b))
            #                 y4 = int(y0 - 3000 * (a))
            #
            #                 if y1 == y2 and y3 == y4:  # Horizontal Lines
            #                     diff = abs(y1 - y3)
            #                 elif x1 == x2 and x3 == x4:  # Vertical Lines
            #                     diff = abs(x1 - x3)
            #                 else:
            #                     diff = 0
            #
            #                 if diff < 200 and diff is not 0:
            #                     continue
            #
            #                 cv2.line(img_undist, (x1, y1), (x2, y2), (255, 0, 0), 1)
            #                 cv2.line(img_undist, (x3, y3), (x4, y4), (255, 0, 0), 1)
            #
            #                 p.append((x1, y1))
            #                 p.append((x2, y2))
            #                 p.append((x3, y3))
            #                 p.append((x4, y4))
            #
            #                 intersectpx, intersectpy = ContourUtils.intersectLines(p[counter], p[counter + 1], p[counter + 2],
            #                                                           p[counter + 3])
            #
            #                 # consider only intersection close to the center of the image
            #                 if (intersectpx < 100 or intersectpx > 800) or (intersectpy < 100 or intersectpy > 800):
            #                     continue
            #
            #                 intersectp.append((intersectpx, intersectpy))
            #
            #                 lines_seg.append([(x1, y1), (x2, y2)])
            #                 lines_seg.append([(x3, y3), (x4, y4)])
            #
            #                 cv2.line(img_undist, (x1, y1), (x2, y2), (255, 0, 0), 1)
            #                 cv2.line(img_undist, (x3, y3), (x4, y4), (255, 0, 0), 1)
            #
            #                 # point offset
            #                 counter = counter + 4
            #     ellipse_vertices.append([(box[1][0] + box[2][0]) / 2, (box[1][1] + box[2][1]) / 2])
            #     ellipse_vertices.append([(box[2][0] + box[3][0]) / 2, (box[2][1] + box[3][1]) / 2])
            #     ellipse_vertices.append([(box[0][0] + box[3][0]) / 2, (box[0][1] + box[3][1]) / 2])
            #     ellipse_vertices.append([(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2])
            #
            #     testpoint1 = M.dot(np.transpose(np.hstack([center_ellipse, 1])))
            #     testpoint2 = M.dot(np.transpose(np.hstack([ellipse_vertices[0], 1])))
            #     testpoint3 = M.dot(np.transpose(np.hstack([ellipse_vertices[1], 1])))
            #     testpoint4 = M.dot(np.transpose(np.hstack([ellipse_vertices[2], 1])))
            #     testpoint5 = M.dot(np.transpose(np.hstack([ellipse_vertices[3], 1])))
            #
            #     newpoints.append([testpoint2[0], testpoint2[1]])
            #     newpoints.append([testpoint3[0], testpoint3[1]])
            #     newpoints.append([testpoint4[0], testpoint4[1]])
            #     newpoints.append([testpoint5[0], testpoint5[1]])
            #     newpoints.append([testpoint1[0], testpoint1[1]])
            #
            #     for lin in lines_seg:
            #         line_p1 = M.dot(np.transpose(np.hstack([lin[0], 1])))
            #         line_p2 = M.dot(np.transpose(np.hstack([lin[1], 1])))
            #         inter1, inter_p1, inter2, inter_p2 = ContourUtils.intersectLineCircle(np.asarray(center_ellipse), circle_radius, np.asarray(line_p1), np.asarray(line_p2))
            #         # cv2.line(image_proc_img, (int(line_p1[0]), int(line_p1[1])), (int(line_p2[0]), int(line_p2[1])), cv.CV_RGB(255, 0, 0), 2, 8)
            #         if inter1:
            #             # cv2.circle(image_proc_img, (int(inter_p1[0]), int(inter_p1[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
            #             inter_p1 = M_inv.dot(np.transpose(np.hstack([inter_p1, 1])))
            #             # cv2.circle(image_proc_img, (int(inter_p1[0]), int(inter_p1[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
            #             if inter2:
            #                 # cv2.circle(image_proc_img, (int(inter_p1[0]), int(inter_p1[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
            #                 inter_p2 = M_inv.dot(np.transpose(np.hstack([inter_p2, 1])))
            #                 # cv2.circle(image_proc_img, (int(inter_p2[0]), int(inter_p2[1])), 3, cv.CV_RGB(0, 0, 255), 2, 8)
            #                 intersectp_s.append(inter_p1)
            #                 intersectp_s.append(inter_p2)
            #
            #         # try:
            #         # calculate mean val between: 0,4;1,5;2,6;3,7
            #         new_intersect = np.mean(([intersectp_s[0], intersectp_s[4]]), axis=0, dtype=np.float32)
            #         points.append(new_intersect)  # top
            #         new_intersect = np.mean(([intersectp_s[1], intersectp_s[5]]), axis=0, dtype=np.float32)
            #         points.append(new_intersect)  # bottom
            #         new_intersect = np.mean(([intersectp_s[2], intersectp_s[6]]), axis=0, dtype=np.float32)
            #         points.append(new_intersect)  # left
            #         new_intersect = np.mean(([intersectp_s[3], intersectp_s[7]]), axis=0, dtype=np.float32)
            #         points.append(new_intersect)  # right
            #     if len(points) != 0:
            #         cv2.circle(img_undist, (int(points[0][0]), int(points[0][1])), 3, (255, 0, 0), 2, 8)
            #         cv2.circle(img_undist, (int(points[1][0]), int(points[1][1])), 3, (255, 0, 0), 2, 8)
            #         cv2.circle(img_undist, (int(points[2][0]), int(points[2][1])), 3, (255, 0, 0), 2, 8)
            #         cv2.circle(img_undist, (int(points[3][0]), int(points[3][1])), 3, (255, 0, 0), 2, 8)

            cv2.imshow("Dart Settings", img_undist)
    cv2.waitKey(1)