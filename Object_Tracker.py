#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import cv2
import Centroid_Tracker as ct
import Ball_Tracker as bt

# Gets distance between ball and cup centroid
def findDis(pts1, pts2):
    x1 = float(pts1[0])
    x2 = float(pts2[0])
    y1 = float(pts1[1])
    y2 = float(pts2[1])
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
    return dis

# Gets Circles from color image
def getCircle(img, imgCont):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.medianBlur(imgGray, 5)
    circles = cv2.HoughCircles(imgBlur,cv2.HOUGH_GRADIENT, 1, 120,param1=130, param2=40, minRadius=0,maxRadius=0)
    inputCentroids = []     # list of cup centroids
    ball = []               # list of ball centroids
    if circles is None:  # if there aren't any circles it returns empty lists
        return imgCont, inputCentroids, ball
    else:
        detected_circles = np.uint16(circles)
        for j in detected_circles[0,:]:
            if j[2] > 40:   # Circles bigger than 40 are cups
                c = [j[0], j[1]]
                inputCentroids.append(c)
                cv2.circle(imgCont, (j[0], j[1]), j[2], (0, 0, 255), 2)
                cv2.circle(imgCont, (j[0], j[1]), 2, (255, 0, 0), 5)
            else:   # Cups less than 40 are balls
                c = [j[0], j[1]]
                ball.append(c)
                cv2.circle(imgCont, (j[0], j[1]), j[2], (0, 0, 255), 2)
                cv2.circle(imgCont, (j[0], j[1]), 2, (255, 0, 0), 5)

        return imgCont, inputCentroids, ball

pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Starts streaming
pipeline.start(config)
cent = ct.CentroidTracker()
bal = bt.BallTracker()
lastBall = []
ID = 0

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    imgCont = color_image.copy()
    imgCircle, inputCentroids, ball = getCircle(color_image, imgCont)
    if len(ball) > 0:   # checks if there are any balls anc record last known centroid
        lastBall = ball
    objects = cent.update(inputCentroids)   # updates cup dictonary
    objectsBall = bal.update(ball)          # updates ball dictonary

    # Displays the Cups on Screen
    if objects is not None:
        for (objectID, centroid) in objects.items():
            text = "CUP {}".format(objectID + 1)
            cv2.putText(imgCircle, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(imgCircle, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.imshow('Circles', imgCircle)

    # Displays the Ball on screen
    if objectsBall is not None:
        for (objectID, centroid) in objectsBall.items():
            text = "Ball {}".format(objectID + 1)
            cv2.putText(imgCircle, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(imgCircle, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            cv2.imshow('Circles', imgCircle)

    # Checks whether ball has vanished and whether it had a last known location
    if len(lastBall) > 0 and len(objectsBall) == 0:
        disList = []   # List of distances between ball and cup centroid
        i = 0
        for (objectID, centroid) in objects.items():
            pts1 = [lastBall[0][0], lastBall[0][1]]
            dis = findDis(pts1, centroid)
            disList.append(dis)
            if i > 0:
                if disList[i] > disList[i - 1]:  # checks if new distance is bigger than previous
                    ID = i
            i = i + 1

        # Continously displays cup which ball is under until a ball appears
        while len(ball) == 0:
            count = 0
            for (objectID, centroid) in objects.items():
                if count == ID:  # Gets centroid of cup ball is under
                    centroidBall = centroid
                else:
                    count = count + 1

            text = "UNDER THIS CUP!"
            cv2.putText(imgCircle, text, (centroidBall[0] - 10, centroidBall[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.circle(imgCircle, (centroidBall[0], centroidBall[1]), 4, (255, 255, 0), -1)
            cv2.imshow('Circles', imgCircle)

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            imgCont = color_image.copy()
            imgCircle, inputCentroids, ball = getCircle(color_image, imgCont)
            objects = cent.update(inputCentroids)
            objectsBall = bal.update(ball)
            cv2.waitKey(1)

        lastBall = []   # Resets last ball location array and ID index
        ID = 0

    cv2.waitKey(1)