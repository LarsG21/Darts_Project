import numpy as np
import cv2
import cv2.aruco as aruco
import os
import utils
import math
import glob
import matplotlib
import matplotlib.pyplot as plt
import random

testing = False
# termination criteria for Subpixel Optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)

scale = 0.2   # Scale Factor for FindCorners in very large images


def calibrateCamera(cap,rows,columns,squareSize,runs,saveImages = False, webcam = True):
    """
    calculates the internal camera parameters of a webcam(live) or an other camera using already taken calibration images
    uses scaled down images for non webcam calibration and searches corners there does the subpixel optimization on the original resolution
    saves all the reprojection errors, K matrix and uncertainty vector in repErrors.txt
    shows parameters and errors using matplotlib
    :param cap: webcam object
    :param rows: rows of the calibration pattern
    :param columns: columns of the calibration pattern
    :param squareSize: square size of the calibration pattern
    :param runs: how many runs of calibrations
    :param saveImages: save calibration images
    :param webcam: if used a webcam or images to calibrate
    :return: mean camera matrix and distortion vector, standard deviation of matrix and vector
    """

    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * squareSize

    directory1 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Vorher"
    directory2 = "C:\\Users\\Lars\\Desktop\\TestBilder\\Nachher"

    open('repErrors.txt', 'w').close()

    print(os.getcwd())
    print('Path Exists ?')

    print(os.path.exists(directory1))
    print(os.path.exists(directory2))
    if not os.path.exists(directory1) or not os.path.exists(directory2):
        saveImages = False
        print("ERROR: Path " + directory1 + " or " + directory2 + " does not exist!")

    allMTX = []
    allDist = []
    allRepErr = []
    if testing:
        fig, ax = plt.subplots()   # Plot for Rep Error
        fig2, ax2 = plt.subplots()  # Plot for K
        fig3, ax3 = plt.subplots()  # Plot for heatmap

    N = 5
    X_Axis = np.arange(N)
    width = 0.35
    meanErrorsBefore = []
    meanErrorsAfter = []

    allpoints = []
    allErrors = []
    first = True

    for r in range(runs):       # for every calibration run
        print('Run ', str(r+1), ' of 5')
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        counter = 0
        images = []

        # reads in Calib Images
        if webcam:  # Read in images from webcam
            while True:
                success, img = cap.read()
                cv2.putText(img, "Press x to take an image of Calicration Pattern. Take at least 10 images from different angles", (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))

                cv2.putText(img, "Run: {:.1f}/5".format(r+1), (210, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
                if counter > 9:
                    cv2.putText(img, "Press q for next step".format(counter), (5, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45,(0, 0, 255))
                else:
                    cv2.putText(img, "Captured: {:.1f}/10".format(counter), (5, 40), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255))
                cv2.imshow("Image", img)

                if cv2.waitKey(1) & 0xff == ord('x'):
                    cv2.putText(img, "Captured", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0))
                    cv2.imshow("Image", img)
                    cv2.waitKey(500)
                    images.append(img)       # save in array
                    if saveImages:
                        utils.saveImagesToDirectory(counter, img, directory1)
                    counter += 1
                    print("Captured")
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
        else:   # Files in Folder

            pathName = "CalibrationImages/Run"+str(r+1)+"/*.TIF"
            images = [cv2.imread(file) for file in glob.glob(pathName)]

        # cv2.imshow("Image", img)
        # cv2.destroyWindow("Image")

        # shows Images
        for frame in images:            # Show Images
            dsize = (1920,1080)
            cv2.imshow("Test",cv2.resize(frame, dsize))
            # cv2.waitKey(20)
        cv2.destroyWindow("Test")

        # findCorners
        counter2 = 0
        MeanErrorDuringOneCalib = []
        PlaceholderList = []
        random.shuffle(images)
        for img in images:
            if not webcam:  # uses a downscaled version of image to give a first guess of corners
                original = img  # keep original
                originalGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                dsize = (int(img.shape[1]*scale), int(img.shape[0]*scale))
                img = cv2.GaussianBlur(img, (3, 3), 1)  # Blur before rezise to avoid alising error
                img = cv2.resize(img, dsize)
            print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            print("Searching for corners...")
            ret, corners = cv2.findChessboardCorners(gray, (columns, rows), None)   # flags=cv2.CALIB_CB_FAST_CHECK   (sorgt aber aurch für false negatives !!!)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print("                        Corners Found")
                objpoints.append(objp)
                if webcam:          # subpixeloptimizer on original image
                    corners2 = cv2.cornerSubPix(gray, corners, (22, 22), (-1, -1), criteria)
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (columns, rows), corners2, ret)
                    if saveImages:
                        utils.saveImagesToDirectory(counter2, img, directory2)
                    counter2 += 1
                    cv2.imshow('img', img)
                else:   # subpixeloptimizer on original image when not webcam
                    corners = corners/scale     # corners must be scaled according to scale factor
                    corners2 = cv2.cornerSubPix(originalGray,corners, (11, 11), (-1, -1), criteria)
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(original, (columns, rows), corners2, ret)
                    if saveImages:
                        utils.saveImagesToDirectory(counter2, img, directory2)
                    counter2 += 1
                    dsize = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                    imgShow = cv2.resize(img, dsize)
                    cv2.imshow('img', imgShow)
                imgpoints.append(corners2)

                # cv2.waitKey(200)       #DELAY
            else:
                print("                        No Corners Found")
            if testing:
                tempret, tempmtx, tempdist, temprvecs, temptvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                                                       gray.shape[::-1], None, None)
                mean_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], temprvecs[i], temptvecs[i], tempmtx, tempdist)
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    mean_error += error
                    if i == 1:
                        pass
                mean_error = mean_error / len(objpoints)
                print("Mean Rep  ERR after image" + str(counter2) + " was: "+str(mean_error))
                MeanErrorDuringOneCalib.append(round(mean_error, 10))
                PlaceholderList.append(counter2)

        # utils.writeLinestoCSV(PlaceholderList,MeanErrorDuringOneCalib,PlaceholderList)

        # message = 'Found Corners in ' +str(counter2) + ' of ' + str(len(images))+ ' images'
        # print('Detect at least 10 for optimal results')
        # print(message)
        dsize = (1920, 1080)
        img = cv2.resize(img, dsize)
        # cv2.putText(img, message, (50, 250), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255),thickness=2)
        cv2.imshow('img', img)
        # cv2.waitKey(2000)                     #DELAY
        cv2.destroyWindow("img")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('Matrix:')
        print(mtx)
        print('Dist:')
        print(dist)
        mean_error = 0
        meanErrorZeroDist = 0
        distZero = np.array([0, 0, 0, 0, 0], dtype=float)
        distCustom = np.array([-0.3635, 0.14126, 0.00209, -0.000267], dtype=float)
        for i in range(len(objpoints)):     # calculate rep Err
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distZero)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            meanErrorZeroDist += error
        meanErrorZeroDist = meanErrorZeroDist / len(objpoints)
        meanErrorsBefore.append(round(meanErrorZeroDist, 10))
        print("Mean error between Ideal Chessboard Corners and Image Corners: {}".format(meanErrorZeroDist))
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, distZero)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)       # imagepoints = corners on images
            for j in range(len(imgpoints2)):                                                # imagepoints2 = reprojected ideal chsessboard corners to image
                if np.linalg.norm(imgpoints2[j]-imgpoints[i][j]) < 100:
                    allErrors.append(np.linalg.norm(imgpoints2[j]-imgpoints[i][j]))
                else:
                    allErrors.append(100)
            if first:
                allpoints = imgpoints2
                first = False
            else:
                allpoints = np.concatenate((allpoints,imgpoints2))
            print(allpoints.shape)
            # print(allpoints)
            # cv2.waitKey(2000)
            print("################################")
            mean_error += error
            if i == 1:
                pass
        mean_error = mean_error / len(objpoints)
        meanErrorsAfter.append(round(mean_error, 10))
        print("Mean error between projected Objectpoints using distortion parameters to Points in real Image: {}".format(mean_error))

        with open('repErrors.txt', 'a') as file:

            books = ["Mean Error before calib in Run {}:".format(r),
                     str(meanErrorZeroDist),
                     "Mean Error after calib in Run {}:".format(r),
                     str(mean_error),
                     "-----------------------------------------------"
                     ]

            file.writelines("% s\n" % data for data in books)
            file.close()
        allMTX.append(mtx)
        allDist.append(dist)
    # ######################  Heatmap  ###############################
    # xi = np.arange(0, 9800, 1)
    # yi = np.arange(0,6600,1)
    # xi, yi = np.meshgrid(xi, yi)

    # mask = (xi > 0.5) & (xi < 0.6) & (yi > 0.5) & (yi < 0.6)

    # zi = griddata((np.array(allpoints[:,:,0]), np.array(allpoints[:,:,1])), np.array(allErrors), (xi, yi), method='linear')

    # zi[mask] = np.nan

    # plt.contourf(xi, yi, zi, np.arange(0, 1.01, 0.01))
    if testing:
        print("max:", max(allErrors))
        ax3.set_title('Reprojection Error Heatmap')
        ax3.set_xlabel(xlabel='X Position (Pixel)')
        ax3.set_ylabel(ylabel='Y Position (Pixel)')
        sc = ax3.scatter(allpoints[:, :, 0], allpoints[:, :, 1], c=allErrors, cmap='turbo', edgecolor='k', marker='+')
        cbar = plt.colorbar(sc, orientation='vertical')
        cbar.ax.set_xlabel("Error (Pixel)")


    # ###################  Plot Projection Errors  ################################
        rects1 = ax.bar(X_Axis, tuple(meanErrorsBefore), width, color='r')
        rects2 = ax.bar(X_Axis+width, tuple(meanErrorsAfter), width, color='g')

        ax.set_ylabel('Reprojectio Error')
        ax.set_title('Before and after Calibration')
        ax.set_xticks(X_Axis + width / 2)
        ax.set_xticklabels(('Run1', 'Run2', 'Run3', 'Run4', 'Run5'))

        ax.legend((rects1[0], rects2[0]), ('before calibration', 'after calibration'))

        plt.draw()
    #####################################################################################
    MTXStack = np.stack(allMTX, axis=1)
    meanMTX = np.mean(MTXStack, axis=1)
    stdMTX = np.std(MTXStack, axis=1)
    print(meanMTX)
    print(stdMTX)

    meanFx = meanMTX[0, 0]
    meanFy = meanMTX[1, 1]
    meanX0 = meanMTX[0, 2]
    meanY0 = meanMTX[1, 2]

    DISTStack = np.stack(allDist, axis=1)
    meanDIST = np.mean(DISTStack, axis=1)
    stdDist = np.std(DISTStack, axis=1)
    print(meanDIST)
    print(stdDist)

    meanK1 = meanDIST[0, 0]
    meanK2 = meanDIST[0, 1]
    meanP1 = meanDIST[0, 2]
    meanP2 = meanDIST[0, 3]
    meanK3 = meanDIST[0, 4]

    # Konfidenzintervall 95% bei 5 Samples T- Verteilung = 1,242

    uncertantyMTX = 1.242*stdMTX
    uncertantyDIST = 1.242*stdDist
    print((meanFx, meanFy, meanX0, meanY0))
    print((uncertantyMTX[0,0], uncertantyMTX[1,1], uncertantyMTX[0,2], uncertantyMTX[1,2]))

    # ####################  Plot Internal Camera Parameters #################################
    if testing:
        rects1 = ax2.bar(np.arange(4), (meanFx, meanFy, meanX0, meanY0), width, color='r', yerr=(uncertantyMTX[0,0], uncertantyMTX[1,1], uncertantyMTX[0,2], uncertantyMTX[1,2]))

        ax2.set_ylabel('[Pixel]')
        ax2.set_title('Internal Camera Parameters')
        ax2.set_xticks(np.arange(4) + width / 2)
        ax2.set_xticklabels(('Fx', 'Fy', 'X0', 'Y0'))
        plt.draw()
    #####################################################################################

    print('Parameter inklusive Konfidenzintervalle (95%):')
    print('fx: ', str(meanFx), ' +/- ', uncertantyMTX[0, 0])
    print('fy: ', str(meanFy), ' +/- ', uncertantyMTX[1, 1])
    print('x0: ', str(meanX0), ' +/- ', uncertantyMTX[0, 2])
    print('y0: ', str(meanY0), ' +/- ', uncertantyMTX[1, 2])
    print('K1: ', str(meanK1), ' +/- ', uncertantyDIST[0, 0])
    print('K2: ', str(meanK2), ' +/- ', uncertantyDIST[0, 1])
    print('P1: ', str(meanP1), ' +/- ', uncertantyDIST[0, 2])
    print('P2: ', str(meanP2), ' +/- ', uncertantyDIST[0, 3])
    print('K3: ', str(meanK3), ' +/- ', uncertantyDIST[0, 4])
    # Wait

    with open('repErrors.txt', 'a') as file:

        books = ["MeanMTX:",
                 str(meanMTX),
                 "uncertaintyMTX:",
                 str(uncertantyMTX),
                 "MeanDist:",
                 str(meanDIST),
                 "uncertaintyDist:",
                 str(uncertantyDIST)
                 ]

        file.writelines("% s\n" % data for data in books)
        file.close()

    # while True:
    #     if cv2.waitKey(1) & 0xff == ord('x'):
    #         break
    #     plt.show()
    cv2.destroyAllWindows()
    return meanMTX, meanDIST, uncertantyMTX, uncertantyDIST
