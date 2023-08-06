import math

import cv2
import numpy as np

MARKER_CORNERS_SAVE, MARKER_IDS_SAVE = None, None

def midpoint(ptA, ptB):
    """
    Calculate the midpoint of A and B
    :param ptA: 2D numpy Array
    :param ptB: 2D numpy Array
    :return: 2D numpy array
    """
    return (ptA[0, 0] + ptB[0, 0]) * 0.5, (ptA[0, 1] + ptB[0, 1]) * 0.5


def get_contours(img, shapeROI = (0, 0), cThr=[100, 150], gaussFilters = 1, dilations = 6, erosions = 2, showFilters=False, minArea=100, epsilon=0.01, Cornerfilter=0, draw=False):
    """
    gets Contours from an image

    :param img: input image (numpy array)
    :param cThr: thrersholds for canny edge detector (list)
    :param gaussFilters: number of gaussian smoothing filters (int)
    :param showFilters: boolean if you want to see the filters
    :param minArea: minimum area of vontours to filter out small noise
    :param epsilon: 'resolution' of polynomial approximation of the contour
    :param Cornerfilter: Only outputs contours with n corners
    :param draw: draws detected contours on img
    :return: image with contours on it, (length of contour, area of contour, poly approximation, boundingbox to the contour, i)
    """
    minArea = minArea/100   # HIGHLIGHT: Only for very small resolution testing
    imgContours = img
    # imgContours = cv2.UMat(img)
    imgGray = cv2.cvtColor(imgContours, cv2.COLOR_BGR2GRAY)
    for i in range(gaussFilters):
       imgGray = cv2.GaussianBlur(imgGray, (11, 11), 1)
    if showFilters:
        cv2.imshow("Gauss", cv2.resize(imgGray.copy(), (int(shapeROI[0]), int(shapeROI[1])), interpolation=cv2.INTER_AREA, fx=0.5, fy=0.5))
    imgCanny = cv2.Canny(imgGray, cThr[0], cThr[1])
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=dilations)
    imgThre = cv2.erode(imgDial, kernel, iterations=erosions)
    if showFilters:
        cv2.imshow('Canny', cv2.resize(imgThre.copy(), (int(shapeROI[0]), int(shapeROI[1])), interpolation=cv2.INTER_AREA, fx=0.5, fy=0.5))
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            # print('minAreaFilled')
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, epsilon * peri, True)
            bbox = cv2.boundingRect(approx)
            if Cornerfilter > 0:
                if len(approx) == Cornerfilter:
                    finalCountours.append([len(approx), area, approx, bbox, i])
            else:
                finalCountours.append([len(approx), area, approx, bbox, i])
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)

    if draw:
        for con in finalCountours:
            cv2.drawContours(imgContours, con[4], -1, (0, 0, 255), 3)

    if not showFilters:
        cv2.destroyWindow("Gauss")
        cv2.destroyWindow("Canny")
    return imgContours, finalCountours, imgThre

def reorder(myPoints):
    """
    Reorders a list of corner points to: top left, top right, bottom left, bottom right
    :param myPoints: list of points (np array)
    :return: reordered points (np array)
    """
    if myPoints.shape == (1, 1, 4, 2):   # 4,1,2
        myPoints = myPoints.reshape(4, 1, 2)
        myPointsNew = np.zeros_like(myPoints)
        myPoints = myPoints.reshape((4, 2))
        # print("RESHAPED_MTX",myPointsNew)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew


def warpImg (img, points, w, h, pad=20):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp


def findDis(pts1, pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5


def extract_roi_from_4_aruco_markers(frame, dsize=(500, 500), draw=False, use_outer_corners=False, hold_position=False):
    """
    This function detects 4 AruCo Markers from the given Library with the IDs 1,2,3,4 (tl,tr,bl,br) and returns the ROI between them
    :param frame: the given frame as np array
    :param dsize: the target size of the returned ROI
    :param draw: If to draw the detected Corners on the frame
    :return: the ROI
    """
    global MARKER_IDS_SAVE, MARKER_CORNERS_SAVE
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    inner_corners = [2, 3, 0, 1]
    if use_outer_corners:
        inner_corners = [0, 1, 2, 3]

    # Detect the markers in the image
    if not hold_position:
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        if draw:
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    else:
        markerCorners = MARKER_CORNERS_SAVE
        markerIds = MARKER_IDS_SAVE
    if markerIds is not None:
        if all(elem in markerIds for elem in [[0], [1], [2], [3]]):
            index = np.squeeze(np.where(markerIds == 0))
            refPt1 = np.squeeze(markerCorners[index[0]])[inner_corners[0]].astype(int)
            index = np.squeeze(np.where(markerIds == 1))
            refPt2 = np.squeeze(markerCorners[index[0]])[inner_corners[1]].astype(int)

            index = np.squeeze(np.where(markerIds == 2))
            refPt3 = np.squeeze(markerCorners[index[0]])[inner_corners[2]].astype(int)
            index = np.squeeze(np.where(markerIds == 3))
            refPt4 = np.squeeze(markerCorners[index[0]])[inner_corners[3]].astype(int)
            h2, status2 = cv2.findHomography(np.asarray([refPt1, refPt2, refPt3, refPt4]), np.asarray([[0, 0], [dsize[0], 0], [dsize[0], dsize[1]], [0, dsize[1]]]))
            warped_image2 = cv2.warpPerspective(frame, h2, dsize)
            # Mark all the Pints
            # save the detected markers to npy file
            MARKER_IDS_SAVE = markerIds
            MARKER_CORNERS_SAVE = markerCorners
            return warped_image2