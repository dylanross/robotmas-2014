#! /usr/bin/env python

import threading
import functools
import time
import cv2
import numpy as np


PRE_SERVO_FRAME_BUFFER_LENGTH = 10
PRE_FIRE_FRAME_BUFFER_LENGTH = 0
FRAME_DELAY = 1
REFRESH_ARRAYS = False


"""
Utility functions.
"""

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



def extend_list(lst, element) :
    try :
        lst.extend(element)
    except TypeError :
        lst.extend([element])


"""
Image processing functions.
"""

def grab_frame() :
    pass


def extract_target(frame) :
    pass


def target_prediction(target_xy_history, xy_prediction_history) :
    pass


"""
Fire control functions.
"""

def firing_decision(target_xy_history, xy_prediction_history) :
    pass


def fire_guns(trig) :
    pass


"""
Motor control functions.
"""

def turning_decision(target_xy_history, xy_prediction_history) :
    pass


def set_servo_position(angle_vec) :
    pass


"""
Bolt the above functions together.
"""

target_xy_history = []
xy_prediction_history = []
angle_history = []
def main_loop(grab_frame=grab_frame, extract_target=extract_target,
        target_prediction=target_prediction,
        firing_decision=firing_decision, fire_guns=fire_guns,
        turning_decision=turning_decision,
        set_servo_position=set_servo_position,
        pre_servo_frame_buffer_length=PRE_SERVO_FRAME_BUFFER_LENGTH,
        pre_fire_frame_buffer_length=PRE_FIRE_FRAME_BUFFER_LENGTH,
        frame_delay=FRAME_DELAY, refresh_arrays=REFRESH_ARRAYS) :

    global target_xy_history, xy_prediction_history, angle_history

    while True :
        # grab new frames
        frame_history = []
        for i in xrange(pre_servo_frame_buffer_length) :
            frame_history.append(grab_frame())
            cv2.imshow('frame', frame_history[-1])
            cv2.waitKey(frame_delay)
        frame_history = np.array(frame_history)

        # extract target x, y coordinates, and add to target x, y history
        target_xy = [extract_target(frame) for frame in frame_history]
        extend_list(target_xy_history, target_xy)

        # predict future target x, y coordinates, and add to prediction history
        pred_xy = target_prediction(target_xy_history, xy_prediction_history)
        extend_list(xy_prediction_history, pred_xy)
    
        # decide how to position servos, act on that decision, wait for servos to update
        angle_vec, servo_latency = turning_decision(target_xy_history, xy_prediction_history)
        angle_history.append(angle_vec)
        set_servo_position(angle_vec)
        time.sleep(servo_latency)

        # grab new frames
        frame_history = []
        for i in xrange(pre_fire_frame_buffer_length) :
            frame_history.append(grab_frame())
            cv2.waitKey(frame_delay)
        frame_history = np.array(frame_history)

        # extract target x, y coordinates, and add to target x, y history
        target_xy = [extract_target(frame) for frame in frame_history]
        extend_list(target_xy_history, target_xy)

        # decide whether to fire, act on that decision
        trig = firing_decision(target_xy_history, xy_prediction_history)
        fire_guns(trig)

        # clear global arrays for fresh start
        if refresh_arrays :
            target_xy_history = []
            xy_prediction_history = []
            angle_history = []
