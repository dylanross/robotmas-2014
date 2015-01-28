#! /usr/bin/env python

import time
import cv2
import numpy as np
import core as CORE

"""
Configure CORE.
"""

CORE.set_cam_resolution((320, 240))


"""
Glocal target extractions parameters.
"""

MOVEMENT_THRESHOLD = 7
FLOW_GAUSSIAN = 5


"""
Target extraction functions.
"""

PREV_FRAME = CORE.grab_frame()
PREV_GRAY = cv2.cvtColor(PREV_FRAME, cv2.COLOR_BGR2GRAY)
HSV = np.zeros_like(PREV_FRAME)
HSV[..., 1] = 255
def extract_target(frame) :
    global PREV_FRAME, PREV_GRAY, HSV, CAM_RES

    # convert to grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(PREV_GRAY, curr_gray, 0.5, 1,
                                        FLOW_GAUSSIAN, 1, 3, 5, 1)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    HSV[...,0] = ang*180/np.pi/2
    HSV[...,2] = mag*10
    rgb = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', rgb)
    
    # threshold and find centroid
    thresh = cv2.threshold(mag.astype(np.uint8), MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # store latest frame as new previous frame
    PREV_FRAME = frame
    PREV_GRAY = curr_gray

    return CORE.target_from_mask(thresh), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR).astype(np.uint8)


TARGET_HSV_CODE = np.array([152.64, 166.055, 68.7675])
TARGET_HSV_CODE = np.array([63.6675, 203.1125, 102.31])
ERROR_BOUNDS = np.array([12, 10, 55])
def color_detect(frame) :
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    H_dist = frame_HSV[:, :, 0] - TARGET_HSV_CODE[0]
    H_dist = 255.*H_dist/float(np.max(H_dist))
    H_dist_thresh = (H_dist > 40).astype(np.uint8)
    cv2.imshow('H distance from target thresholded', 255*H_dist_thresh)

    S_dist = frame_HSV[:, :, 1] - TARGET_HSV_CODE[1]
    S_dist = 255.*S_dist/float(np.max(S_dist))
    S_dist_thresh = (S_dist > 40).astype(np.uint8)
    cv2.imshow('S distance from target thresholded', 255*S_dist_thresh)

    target_mask = 255*H_dist_thresh*S_dist_thresh
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, np.ones((2, 2)))
    cv2.imshow('H * S thresholds', target_mask)

    #color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, np.ones((3, 3)))

    return CORE.target_from_mask(target_mask), cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)


"""
Test functions.
"""

def test_real_time_processing(proc_func=color_detect) :
    while True :
        frame = CORE.grab_frame()
        target_xy, proc_frame = proc_func(frame)

        if target_xy[0] == None : target_xy[0] = 0
        if target_xy[1] == None : target_xy[1] = 0
        trg = (target_xy[0] + CORE.CAM_CENTRE[0], target_xy[1] + CORE.CAM_CENTRE[1])
        CORE.draw_target(frame, trg)
        CORE.draw_target(proc_frame, trg)

        cv2.imshow('CAM UNPROCESSED', frame)
        cv2.imshow('CAM PROCESSED', proc_frame)

        cv2.waitKey(1)


def capture_processing(proc_func=color_detect, dur=2) :
    frames = []
    target_xy_hist = []
    proc_frames = []
    t0 = time.time()
    while time.time() - t0 <= dur :
        frame = CORE.grab_frame()
        frames.append(frame)
        target_xy, proc_frame = proc_func(frame)
        target_xy_hist.append(target_xy)
        proc_frames.append(proc_frames)

        if target_xy[0] == None : target_xy[0] = 0
        if target_xy[1] == None : target_xy[1] = 0
        trg = (target_xy[0] + CORE.CAM_CENTRE[0], target_xy[1] + CORE.CAM_CENTRE[1])
        CORE.draw_target(frame, trg)
        CORE.draw_target(proc_frame, trg)

        cv2.imshow('CAM UNPROCESSED', frame)
        cv2.imshow('CAM PROCESSED', proc_frame)

        cv2.waitKey(1)

    return frames, proc_frames, target_xy_hist
