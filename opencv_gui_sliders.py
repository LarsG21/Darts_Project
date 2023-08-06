import cv2


def updateTrackBar():
    """
    updates the trackbars
    :return: the trackbar positions
    """
    cannyLow = cv2.getTrackbarPos("Edge Thresh Low", "Edge Detection Settings")
    cannyHigh = cv2.getTrackbarPos("Edge Thresh High", "Edge Detection Settings")
    noGauss = cv2.getTrackbarPos("Gaussian's", "Edge Detection Settings")
    dilations = cv2.getTrackbarPos("Dilations","Edge Detection Settings")
    erosions = cv2.getTrackbarPos("Erosions", "Edge Detection Settings")
    minArea = cv2.getTrackbarPos("minArea", "Edge Detection Settings")
    epsilon = (cv2.getTrackbarPos("Epsilon", "Edge Detection Settings")) / 1000
    showFilters = bool(cv2.getTrackbarPos("Show Filters", "General Settings"))
    automatic = bool(cv2.getTrackbarPos("Automatic", "General Settings"))
    textsize = cv2.getTrackbarPos("TextSize", "General Settings") / 100

    return cannyLow, cannyHigh, noGauss, minArea, erosions, dilations, epsilon, showFilters, automatic, textsize


def update_dart_trackbars():
    radius_1 = cv2.getTrackbarPos("Circle1", "Dart Settings")
    radius_2 = cv2.getTrackbarPos("Circle2", "Dart Settings")
    radius_3 = cv2.getTrackbarPos("Circle3", "Dart Settings")
    radius_4 = cv2.getTrackbarPos("Circle4", "Dart Settings")
    radius_5 = cv2.getTrackbarPos("Circle5", "Dart Settings")
    radius_6 = cv2.getTrackbarPos("Circle6", "Dart Settings")
    x_offset = cv2.getTrackbarPos("X_Offset", "Dart Settings")
    y_offset = cv2.getTrackbarPos("Y_Offset", "Dart Settings")

    return radius_1, radius_2, radius_3, radius_4, radius_5, radius_6, x_offset, y_offset


def resetTrackBar():
    """
    resets trackbars to default values
    """
    cv2.setTrackbarPos("Edge Thresh Low", "Edge Detection Settings", 120)
    cv2.setTrackbarPos("Edge Thresh High", "Edge Detection Settings", 160)
    cv2.setTrackbarPos("Gaussian's", "Edge Detection Settings", 2)
    cv2.setTrackbarPos("Dilations", "Edge Detection Settings", 6)
    cv2.setTrackbarPos("Erosions", "Edge Detection Settings", 2)
    cv2.setTrackbarPos("minArea", "Edge Detection Settings", 800)
    cv2.setTrackbarPos("Epsilon", "Edge Detection Settings", 100)


def create_gui():
    mainImage = cv2.imread("Resources/Main Frame.PNG")
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
    cv2.createTrackbar("Show Filters", "General Settings", 0, 1, empty)
    cv2.createTrackbar("Automatic", "General Settings", 0, 1, empty)
    cv2.createTrackbar("TextSize", "General Settings", 32, 100, empty)
    cv2.createTrackbar("Circle1", "Dart Settings", 4, 100, empty)
    cv2.createTrackbar("Circle2", "Dart Settings", 8, 100, empty)
    cv2.createTrackbar("Circle3", "Dart Settings", 42, 100, empty)
    cv2.createTrackbar("Circle4", "Dart Settings", 47, 100, empty)
    cv2.createTrackbar("Circle5", "Dart Settings", 70, 100, empty)
    cv2.createTrackbar("Circle6", "Dart Settings", 74, 100, empty)
    # cv2.createTrackbar("Circle1", "Dart Settings", 28, 100, empty)
    # cv2.createTrackbar("Circle2", "Dart Settings", 10, 100, empty)
    # cv2.createTrackbar("Circle3", "Dart Settings", 58, 100, empty)
    # cv2.createTrackbar("Circle4", "Dart Settings", 66, 100, empty)
    # cv2.createTrackbar("Circle5", "Dart Settings", 94, 100, empty)
    # cv2.createTrackbar("Circle6", "Dart Settings", 100, 100, empty)
    cv2.createTrackbar("X_Offset", "Dart Settings", 0, 100, empty)
    cv2.setTrackbarMin("X_Offset", "Dart Settings", -100)
    cv2.createTrackbar("Y_Offset", "Dart Settings", 0, 100, empty)
    cv2.setTrackbarMin("Y_Offset", "Dart Settings", -100)

    cv2.destroyWindow("Edge Detection Settings")
    cv2.destroyWindow("General Settings")
