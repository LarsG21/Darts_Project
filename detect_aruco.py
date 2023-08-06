import cv2


dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()

cap = cv2.VideoCapture(1)


while True:
    success, img = cap.read()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    img = cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds)
    print(markerIds)
    cv2.imshow("Image", img)
    cv2.waitKey(1)