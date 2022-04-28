import cv2


def updateTrackBar():
    """
    updates the trackbars
    :return:
    """
    cannyLow = cv2.getTrackbarPos("Edge Thresh Low", "Edge Detection Settings")
    cannyHigh = cv2.getTrackbarPos("Edge Thresh High", "Edge Detection Settings")
    noGauss = cv2.getTrackbarPos("Gaussian's", "Edge Detection Settings")
    dialations = cv2.getTrackbarPos("Dilations","Edge Detection Settings")
    errosions = cv2.getTrackbarPos("Erosions", "Edge Detection Settings")
    minArea = cv2.getTrackbarPos("minArea", "Edge Detection Settings")
    epsilon = (cv2.getTrackbarPos("Epsilon", "Edge Detection Settings")) / 1000
    showFilters = bool(cv2.getTrackbarPos("Show Filters", "General Settings"))
    automatic = bool(cv2.getTrackbarPos("Automatic", "General Settings"))
    textsize = cv2.getTrackbarPos("TextSize", "General Settings")/100

    return cannyLow, cannyHigh, noGauss, minArea, errosions, dialations, epsilon, showFilters, automatic, textsize


def update_dart_trackbars():
    radius_1 = cv2.getTrackbarPos("Circle1", "Dart Settings")
    radius_2 = cv2.getTrackbarPos("Circle2", "Dart Settings")
    radius_3 = cv2.getTrackbarPos("Circle3", "Dart Settings")
    x_offset = cv2.getTrackbarPos("X_Offset", "Dart Settings")
    y_offset = cv2.getTrackbarPos("Y_Offset", "Dart Settings")

    return radius_1, radius_2, radius_3, x_offset, y_offset


def resetTrackBar():
    """
    resets trackbars to default values
    :return:
    """
    cv2.setTrackbarPos("Edge Thresh Low", "Edge Detection Settings", 120)
    cv2.setTrackbarPos("Edge Thresh High", "Edge Detection Settings", 160)
    cv2.setTrackbarPos("Gaussian's", "Edge Detection Settings", 2)
    cv2.setTrackbarPos("Dilations", "Edge Detection Settings", 6)
    cv2.setTrackbarPos("Erosions", "Edge Detection Settings", 2)
    cv2.setTrackbarPos("minArea", "Edge Detection Settings", 800)
    cv2.setTrackbarPos("Epsilon", "Edge Detection Settings", 100)
