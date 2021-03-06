from statistics import mode
from time import sleep

import pickle
from FPS import FPS

import CalibrationWithUncertainty
import ContourUtils
from Dart_Scoring.dart_scorer_util import update_score
from utils import rez, reset_default_image
import gui
from CalibrationWithUncertainty import *

from Dart_Scoring import dart_scorer_util, DartScore

# Globals
points = []
intersectp = []
ellipse_vertices = []
newpoints = []
intersectp_s = []
dart_point = None
TRIANGLE_DETECT_THRESH = 11
useMovingAverage = False
score1 = DartScore.Score(501, True)
scored_values = []
scored_mults = []

values_of_round = []
mults_of_round = []


# #############  Config  ####################
saveImages = False
saveParametersPickle = False
loadSavedParameters = True
webcam = True
rows = 6  # 17   6
columns = 9  # 28    9
squareSize = 30  # mm
calibrationRuns = 1
CAMERA_NUMBER = 0  # 0,1 is built-in, 2 is external webcam


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

#####################################################################################

target_ROI_size = (600, 600)
previous_img = np.zeros((target_ROI_size[0], target_ROI_size[1], 3)).astype(np.uint8)
difference = np.zeros(target_ROI_size).astype(np.uint8)



cv2.destroyWindow("Object measurement")

# First While Loop to set the default reference img
resize_for_squish = (600, 600)
def set_default_img(cap):
    default_img_temp = np.zeros(target_ROI_size).astype(np.uint8)
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
                default_img_temp = img_roi
                print("Set default image")
                cv2.imshow("Default", default_img_temp)
                cv2.waitKey(1)
            if cv2.waitKey(1) & 0xff == ord('x'):
                cv2.destroyWindow("Preview")
                cv2.destroyWindow("Default")
                return default_img_temp

default_img = set_default_img(cap=cap)

# cv2.destroyWindow("General Settings")
# cv2.destroyWindow("Edge Detection Settings")

images_for_rolling_average = []
ellipse = None
x_offset_current, y_offset_current = 0, 0


def detect_dart_circle_and_set_limits():
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


while True:
    fpsReader = FPS()
    success, img = cap.read()
    if success:
        img_undist = utils.undistortFunction(img, meanMTX, meanDIST)
        # cv2.imshow("Undist", img_undist)
        img_roi = ContourUtils.extract_roi_from_4_aruco_markers(img_undist, target_ROI_size, use_outer_corners=False)
        if img_roi is not None and img_roi.shape[1] > 0 and img_roi.shape[0] > 0:
            img_roi = cv2.resize(img_roi, resize_for_squish)
            cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automaticMode, threshold_new = gui.updateTrackBar()

            detect_dart_circle_and_set_limits()

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
                        pt1, pt2, pt3 = np.array([-1,-1]), np.array([-1,-1]), np.array([-1,-1])
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

                    if len(scored_values) <= 20:
                        scored_values.append(new_val)
                        scored_mults.append(new_mult)
                    else:
                        final_val = mode(scored_values)
                        final_mult = mode(scored_mults)
                        values_of_round.append(final_val)
                        mults_of_round.append(final_mult)
                        default_img = reset_default_image(img_undist, target_ROI_size, resize_for_squish)
                        print(f"Final val {final_val}")
                        print(f"Final mult {final_mult}")
                        if len(values_of_round) == 3:
                            update_score(score1, values_of_round=values_of_round, mults_of_round=mults_of_round)
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
            cv2.imshow("Dart Settings", rez(img_roi,1.5))
        else:
            print("NO MARKERS FOUND!")
            cv2.putText(img_undist, "NO MARKERS FOUND", (300, 300), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)
            cv2.imshow("Dart Settings", img_undist)
            sleep(1)

        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cap.release()
            exit()
