#! /usr/bin/env python

import cv2
import numpy as np

CAM0_ID = 0
CAM0_RES = (320, 240)
CAM1_ID = 1
CAM1_RES = (320, 240)

CAM0 = cv2.VideoCapture(CAM0_ID)
CAM0.set(3, CAM0_RES[0])
CAM0.set(4, CAM0_RES[1])

CAM1 = cv2.VideoCapture(CAM1_ID)
CAM1.set(3, CAM1_RES[0])
CAM1.set(4, CAM1_RES[1])


def test_cam() :
    while True :
        frame0 = CAM0.read()[1]
        frame1 = CAM1.read()[1]

        cv2.imshow('cam0', frame0)
        cv2.imshow('cam1', frame1)

        cv2.waitKey(1)


def terminate() :
    CAM0.release()
    CAM1.release()
    cv2.destroyAllWindows()
