import cv2
# from cv2 import aruco
import argparse
import sys
import os.path
import numpy as np
import ContourUtils



cap = cv2.VideoCapture(0)

# Get the video writer initialized to save the output video
# if (not args.image):
#     vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28, (round(2 * cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

im_src = cv2.imread("new_scenery.jpg")

winName = "Augmented Reality using Aruco markers in OpenCV"



while cv2.waitKey(1) < 0:
    # try:
        # get frame from the video
        hasFrame, frame = cap.read()

        if hasFrame:
            ###########################################USE THE MODULE################################################
            warp = ContourUtils.extract_roi_from_4_aruco_markers(frame,dsize=(600,600),draw=True,use_outer_corners=True)
            if warp is not None and warp.shape[1] >0 and warp.shape[0] > 0:
                cv2.imshow("Warp",warp)
                cv2.waitKey(1)
            cv2.imshow("Frame",frame)
            ######################################MANUAL CODE WITH MORE DETAIL###################################
        #     # Load the dictionary that was used to generate the markers.
        #     dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        #
        #     # Initialize the detector parameters using default values
        #     parameters = cv2.aruco.DetectorParameters_create()
        #
        #     # Detect the markers in the image
        #     markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        #
        #     # print(markerIds)
        #     # print(markerCorners)
        #
        #     if markerIds is not None:
        #         cv2.aruco.drawDetectedMarkers(frame, markerCorners)  # Drwas Box around Marker
        #
        #         if all(elem in markerIds for elem in [[0], [1], [2], [3]]):
        #             # print("All in there")
        #             index = np.squeeze(np.where(markerIds == 0))
        #             refPt1 = np.squeeze(markerCorners[index[0]])[2].astype(int)
        #             index = np.squeeze(np.where(markerIds == 1))
        #             refPt2 = np.squeeze(markerCorners[index[0]])[3].astype(int)
        #             distance = np.linalg.norm(refPt1 - refPt2)
        #
        #             scalingFac = 0.02
        #             pts_dst = [[refPt1[0] - round(scalingFac * distance), refPt1[1] - round(scalingFac * distance)]]
        #             pts_dst = pts_dst + [[refPt2[0] + round(scalingFac * distance), refPt2[1] - round(scalingFac * distance)]]
        #
        #             index = np.squeeze(np.where(markerIds == 2))
        #             refPt3 = np.squeeze(markerCorners[index[0]])[0].astype(int)
        #             pts_dst = pts_dst + [[refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)]]
        #             index = np.squeeze(np.where(markerIds == 3))
        #             refPt4 = np.squeeze(markerCorners[index[0]])[1].astype(int)
        #             pts_dst = pts_dst + [[refPt4[0] - round(scalingFac * distance), refPt4[1] + round(scalingFac * distance)]]
        #
        #             pts_src = [[0, 0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]]
        #             print(pts_dst)
        #
        #             #Mark all the Pints
        #             for point in [refPt1,refPt2,refPt3,refPt4]:
        #                 cv2.circle(frame,point,3,(255,0,255),4)
        #
        #             pts_src_m = np.asarray(pts_src)
        #             pts_dst_m = np.asarray(pts_dst)
        #
        #             #Dummy np array
        #             dummy = np.zeros((500, 500), dtype=np.uint8)
        #             h2, status2 = cv2.findHomography(np.asarray([refPt1,refPt2,refPt3,refPt4]), np.asarray([[0, 0], [500,0], [500, 500], [0, 500]]))
        #             warped_image2 = cv2.warpPerspective(frame, h2, (500, 500))
        #             cv2.imshow("Warp2", warped_image2)
        #
        #             # Calculate Homography
        #             h, status = cv2.findHomography(pts_src_m, pts_dst_m)
        #
        #             # Warp source image to destination based on homography
        #             warped_image = cv2.warpPerspective(im_src, h, (frame.shape[1], frame.shape[0]))
        #
        #             cv2.imshow("Warp",warped_image)
        #
        #             # Prepare a mask representing region to copy from the warped image into the original frame.
        #             mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
        #             cv2.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv2.LINE_AA)
        #
        #             # Erode the mask to not copy the boundary effects from the warping
        #             element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #             mask = cv2.erode(mask, element, iterations=3)
        #
        #             cv2.imshow("Mask",mask)
        #
        #
        #             # Copy the mask into 3 channels.
        #             warped_image = warped_image.astype(float)
        #             mask3 = np.zeros_like(warped_image)
        #             for i in range(0, 3):
        #                 mask3[:, :, i] = mask / 255
        #
        #             # Copy the warped image into the original frame in the mask region.
        #             warped_image_masked = cv2.multiply(warped_image, mask3)
        #             frame_masked = cv2.multiply(frame.astype(float), 1 - mask3)
        #             im_out = cv2.add(warped_image_masked, frame_masked)
        #
        #             # Showing the original image and the new output image side by side
        #             concatenatedOutput = cv2.hconcat([frame.astype(float), im_out])
        #             cv2.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))
        #
        #     # Write the frame with the detection boxes
        #     # if (args.image):
        #     #     cv2.imwrite(outputFile, concatenatedOutput.astype(np.uint8))
        #     # else:
        #     #     vid_writer.write(concatenatedOutput.astype(np.uint8))
        #
        # cv2.imshow("Markers", frame)

cv2.destroyAllWindows()
