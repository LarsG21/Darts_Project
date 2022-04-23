import math

import cv2
import numpy as np

def midpoint(ptA, ptB):
    """
    Calculate the midpoint of A and B
    :param ptA: 2D numpy Array
    :param ptB: 2D numpy Array
    :return: 2D numpy array
    """
    return ((ptA[0,0] + ptB[0,0]) * 0.5, (ptA[0,1] + ptB[0,1]) * 0.5)


def get_contours(img, shapeROI = (0,0), cThr=[100, 150], gaussFilters = 1,dialations = 6,errsoions = 2, showFilters=False, minArea=100, epsilon = 0.01, Cornerfilter=0, draw=False):
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
    minArea = minArea/100   #HIGHLIGHT: Only for very small resolution testing
    imgContours = img
    #imgContours = cv2.UMat(img)
    imgGray = cv2.cvtColor(imgContours, cv2.COLOR_BGR2GRAY)
    for i in range(gaussFilters):
       imgGray = cv2.GaussianBlur(imgGray, (11, 11),1)
    if showFilters: cv2.imshow("Gauss",cv2.resize(imgGray, (int(shapeROI[0]),int(shapeROI[1])), interpolation=cv2.INTER_AREA,fx=0.5,fy=0.5))
    imgCanny = cv2.Canny(imgGray, cThr[0], cThr[1])
    kernel = np.ones((3, 3))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=dialations)
    imgThre = cv2.erode(imgDial, kernel, iterations=errsoions)
    if showFilters: cv2.imshow('Canny',cv2.resize(imgThre, (int(shapeROI[0]),int(shapeROI[1])), interpolation=cv2.INTER_AREA,fx=0.5,fy=0.5))
    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            #print('minAreaFilled')
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
    if myPoints.shape == (1,1,4,2):   #4,1,2
        myPoints = myPoints.reshape(4,1,2)
        myPointsNew = np.zeros_like(myPoints)
        myPoints = myPoints.reshape((4,2))
        #print("RESHAPED_MTX",myPointsNew)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints,axis=1)
        myPointsNew[1]= myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

def warpImg (img,points,w,h,pad=20):
    # print(points)
    points =reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

def intersectLines(pt1, pt2, ptA, ptB):
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)

        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1
    x2, y2 = pt2
    dx1 = x2 - x1
    dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA
    xB, yB = ptB
    dx = xB - x
    dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if math.fabs(DET) < DET_TOLERANCE:
        return 0, 0

    # now, the determinant should be OK
    DETinv = 1.0 / DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy * (x - x1) + dx * (y - y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x - x1) + dx1 * (y - y1))

    # return the average of the two descriptions
    x = (x1 + r * dx1 + x + s * dx) / 2.0
    y = (y1 + r * dy1 + y + s * dy) / 2.0
    return x, y

def intersectLineCircle(center, radius, p1, p2):
    baX = p2[0] - p1[0]
    baY = p2[1] - p1[1]
    caX = center[0] - p1[0]
    caY = center[1] - p1[1]

    a = baX * baX + baY * baY
    bBy2 = baX * caX + baY * caY
    c = caX * caX + caY * caY - radius * radius

    pBy2 = bBy2 / a
    q = c / a

    disc = pBy2 * pBy2 - q
    if disc < 0:
        return False, None, False, None

    tmpSqrt = math.sqrt(disc)
    abScalingFactor1 = -pBy2 + tmpSqrt
    abScalingFactor2 = -pBy2 - tmpSqrt

    pint1 = p1[0] - baX * abScalingFactor1, p1[1] - baY * abScalingFactor1
    if disc == 0:
        return True, pint1, False, None

    pint2 = p1[0] - baX * abScalingFactor2, p1[1] - baY * abScalingFactor2
    return True, pint1, True, pint2

