import cv2
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

for id in range(4):
    img = aruco.drawMarker(aruco_dict, id=id, sidePixels=140)
    cv2.imshow("Aruco", img)
    cv2.imwrite(f"ArucoID{id}.png", img)
    cv2.waitKey(100)
