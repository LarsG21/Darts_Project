import cv2
import os
import ContourUtils
import csv
from datetime import datetime
from scipy.spatial import distance as dist

def saveImagesToDirectory(counter,img,directory):
    """
    Saves an image to an given directory using a counter
    :param counter: counter needs to be increased for every new image to avoid overwriting
    :param img: the image (np array)
    :param directory: the desired directory (string)
    :return: none
    """
    owd = os.getcwd()       #save original directory
    if os.path.exists(directory):
        os.chdir(directory)     #change working directory
    else:
        print("ERROR: Directory not Found")
    filename = 'savedImage' + str(counter) + '.TIF'
    cv2.imwrite(filename, img)  # in Ordner Speichern
    print(counter)
    os.chdir(owd)       #go back to original directory


def saveFileToDirectory(filename,filetype,file,directory):
    if os.path.exists(directory):
        os.chdir(directory)
    else:
        print("ERROR: Directory not Found")
    name = filename + '.' + filetype
    cv2.imwrite(name, file)  # in Ordner Speichern
    print("Saved",name)



def undistortFunction(img,mtx,dist):
    """
    undistorts an image given the camera matrix and distortion coefficients
    :param img: the image to undistort
    :param mtx: camera matrix (3x3) numpy array
    :param dist: distortion vector (1x5) numpy array [k1,k2,p1,p2,k3]
    :return: the undistorted image
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    #cv2.imwrite('calibresult.png',dst)
    return dst

def calculatePixelsPerMetric(img,reorderd,ArucoSize,draw = True):
    """
    Calculates the pixels/mm in a given image with an AruCo marker (in the plane of the marker)
    :param img: image with AruCo marker
    :param reorderd: the reordered corner points of the marker
    :param ArucoSize: the size of the marker in mm
    :param draw: bool if the individual pixels/mm for x and y direction should be drawn on screen
    :return: returns the average of pixels/mm in x and y direction
    """

    (tltrX, tltrY) = ContourUtils.midpoint(reorderd[0], reorderd[1])  # top left,top right
    (blbrX, blbrY) = ContourUtils.midpoint(reorderd[2], reorderd[3])  # bottom left, botto right
    (tlblX, tlblY) = ContourUtils.midpoint(reorderd[0], reorderd[2])
    (trbrX, trbrY) = ContourUtils.midpoint(reorderd[1], reorderd[3])
    if draw:
        cv2.circle(img, (int(tltrX), int(tltrY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(blbrX), int(blbrY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(tlblX), int(tlblY)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(trbrX), int(trbrY)), 1, (255, 255, 0), 2)
        cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),  # Draws Lines in Center
                 (255, 0, 255), 2)
        cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                 (255, 0, 255), 2)

    dY = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dX = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    pixelsPerMetric = (dX + dY) / 2 * (1 / ArucoSize)  # Calculates Pixels/Length Parameter
    dimA = dY / pixelsPerMetric
    dimB = dX / pixelsPerMetric  # Dimention of Marker
    if draw:
        cv2.putText(img, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
        cv2.putText(img, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2)
    return pixelsPerMetric




def undistortPicture(cap,saveImages,meanMTX,meanDIST):
    """
    takes an image from a webcam cap and undistorts it, optionally saves image
    :param cap: webcam object
    :param saveImages: bool if save or not
    :param meanMTX: camera matrix (3x3) numpy array
    :param meanDIST: distortion vector (1x5) numpy array [k1,k2,p1,p2,k3]
    :return: none
    """
    print("Take picture to undistort")
    while True:
        succsess, img = cap.read()
        cv2.imshow("Calib_Chess", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        if cv2.waitKey(1) & 0xff == ord('x'):
            succsess,image = cap.read()
            cv2.imshow("Distorted",image)
            if saveImages:
                saveImagesToDirectory("_distorted",image, "C:\\Users\\Lars\\Desktop\\TestBilder")
            undist = undistortFunction(image,meanMTX,meanDIST)
            cv2.imshow("Undistorted",undist)
            if saveImages:
                saveImagesToDirectory("_undistorted",undist, "C:\\Users\\Lars\\Desktop\\TestBilder")
            cv2.waitKey(2000)
        cv2.waitKey(1)

def cropImage(im):
    """
    crops an image with a user selected ROI
    :param im: image to crop
    :return: the croped image
    """
    # Read image
    scale = 0.15
    imcopy = im.copy()
    resized = cv2.resize(imcopy,(int(imcopy.shape[1]*scale),int(imcopy.shape[0]*scale)))  #select in scaled image
    # Select ROI
    r = cv2.selectROI(resized)

    # Crop image
    imCrop = im[int(r[1]/scale):int((r[1] + r[3])/scale), int(r[0]/scale):int((r[0] + r[2])/scale)]  #acually resize the original
    shape = imCrop.shape

    # Display cropped image
    return imCrop, shape

def writeLinestoCSV(startPointList, endPointList, distanceList):
    """
    Writes lists of startingpoints distances and endpoints to a csv file
    :param startPointList: lists of startingpoints
    :param endPointList: lists of endpoints
    :param distanceList: lists of distances
    :return:
    """
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open('Results/'+filename+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(["point1", "distance", "point2"])
        for (point1,distance,point2) in zip(startPointList,distanceList,endPointList):
            spamwriter.writerow([point1,round(distance,6),point2])