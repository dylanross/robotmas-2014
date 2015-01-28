#! /usr/bin/env python

import time
import threading
import functools
import cv2
import numpy as np
from pyfirmata import Arduino


ARDUINO_PORT = '/dev/ttyACM0'
print('Connecting to arduino at ' + str(ARDUINO_PORT) + '...')
BOARD = Arduino(ARDUINO_PORT)
print('Connected OK!')

print('Configuring servo control...')
THETA_PIN = BOARD.get_pin('d:3:p')
THETA_MIN = 0.0
THETA_MAX = 105.0
THETA_PWM_MIN = 0.40
THETA_PWM_MAX = 0.99

PHI_PIN = BOARD.get_pin('d:9:p')
PHI_MIN = 0.0
PHI_MAX = 105.0
PHI_PWM_MIN = 0.40
PHI_PWM_MAX = 0.99
print('Configuration OK!')

print('Configuring gun control...')
GUN0_PIN = BOARD.get_pin('d:12:o')
GUN1_PIN = BOARD.get_pin('d:8:o')
GUN_DELAY = 1./6.
BURST_MODE = True
BURST_LENGTH = 1.0
FIRING_RADIUS = 100.0
print('Configuration OK!')

CAM0_ID = 0
print('Connecting to camera, ID=' + str(CAM0_ID) + '...')
CAM = cv2.VideoCapture(CAM0_ID)
print('Connected OK!')
print('Configuring camera, ID=' + str(CAM0_ID) + '...')
CAM_RES = (640, 480)
CAM_CENTRE = (int(CAM_RES[0]/2.), int(CAM_RES[1]/2.))
CAM.set(3, CAM_RES[0])
CAM.set(4, CAM_RES[1])
print('Configuration OK!')

print('Configuring machine vision algorithms...')
THRESHOLD = 70
NOISE_THRESHOLD = 1000000
print('Configuration OK!')


def run_async(func):
    """
    Function decorator intended to make "func" run in a separate thread
    (asynchronously). Thread object will be returned.
    """

    @functools.wraps(func)
    def async_func(*args, **kwargs):
	func_hl = threading.Thread(target=func, args=args, kwargs=kwargs)
	func_hl.start()
	return func_hl

    return async_func


def position_command(mvec) :
    """
    Set position of theta (azimuth) and phi (elevation) servomotors. Input
    should be a two element list-like object, mvec=[theta, phi], with theta and
    phi in the range 0..1. This value is internally scaled to lie between the
    minimum and maximum PWM outputs specified by global variables
    THETA_PWM_MIN, THETA_PWM_MAX, PHI_PWM_MIN, and PHI_PWM_MAX. This function
    returns nothing.
    """
    global THETA_PIN, THETA_PWM_MAX, THETA_PWM_MIN
    global PHI_PIN, PHI_PWM_MAX, PHI_PWM_MIN

    theta = (THETA_PWM_MAX - THETA_PWM_MIN)*mvec[0] + THETA_PWM_MIN
    phi = (PHI_PWM_MAX - PHI_PWM_MIN)*mvec[1] + PHI_PWM_MIN
    THETA_PIN.write(theta)
    PHI_PIN.write(phi)


print('Homing servomotors...')
position_command([0.0, 0.0])
print('Homing OK!')


@run_async
def gun_command(trig) :
    """
    Control gun triggers. 
    """
    global GUN0_PIN, GUN1_PIN, GUN_PERIOD

    trig = bool(trig)

    if trig == True :
        GUN0_PIN.write(1)
        time.sleep(GUN_DELAY)
        GUN1_PIN.write(1)

        if BURST_MODE == True :
            time.sleep(BURST_LENGTH)
            GUN0_PIN.write(0)
            GUN1_PIN.write(0)

    elif trig == False :
        GUN0_PIN.write(0)
        GUN1_PIN.write(0)

        
def test_cam() :
    CURRENTLY_FIRING = False
    while True :
        # grab frame, convert to grayscale, threshold, set to zero if below noise threshold
        frame = CAM.read()[1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
        thresh_sum = np.sum(thresh)

        # compute centroid, make firing decision, move servos
        M0 = cv2.moments(thresh)
        try :
            if thresh_sum >= NOISE_THRESHOLD :
                cx = int(M0['m10']/M0['m00'])
                cy = int(M0['m01']/M0['m00'])
            else : 
                cx = 0
                cy = 0 
        except ZeroDivisionError :
            cx = 0
            cy = 0

        cx_rel = cx - CAM_CENTRE[0]
        cy_rel = cy - CAM_CENTRE[1]

        if np.linalg.norm([cx_rel, cy_rel]) <= FIRING_RADIUS and CURRENTLY_FIRING == False and thresh_sum >= NOISE_THRESHOLD :
            print('FIRE!!!')
            gun_command(True)
            CURRENTLY_FIRING = True

        elif np.linalg.norm([cx_rel, cy_rel]) >= FIRING_RADIUS and CURRENTLY_FIRING == True and thresh_sum >= NOISE_THRESHOLD :
            print('acquiring target...')
            gun_command(False)
            CURRENTLY_FIRING = False

        position_command([float(cx)/CAM_RES[0], float(cy)/CAM_RES[1]])

        # present graphical representation to user
        thresh_clr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.circle(thresh_clr, (cx, cy), 5, (0, 0, 255), thickness=-1)
        cv2.circle(thresh_clr, CAM_CENTRE, int(FIRING_RADIUS), (0, 100, 0), thickness=1)

        cv2.imshow('cam0', frame)
        cv2.imshow('cam0 threshold', thresh_clr)

        cv2.waitKey(1)


def terminate() :
    CAM.release()
    cv2.destroyAllWindows()
