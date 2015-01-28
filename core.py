#! /usr/bin/env python

import time
import threading
import functools
import cv2
import numpy as np
from pyfirmata import Arduino


"""
Global parameter declarations.
"""

# UI parameters
VERBOSITY = 1                           # max. verbosity level of messages
                                        # to be shown to user (see vprint())

# System parameters
MOVE_ENABLED = False                    # enable turret movement
FIRE_ENABLED = True                     # enable turret firing
GUI_ENABLED = True                      # enable graphical user interface

# Arduino
ARDUINO_PORT = '/dev/ttyACM0'           # port arduino is connected to
BOARD = Arduino(ARDUINO_PORT)           # object representing arduino

# Azimuth servo control
THETA_PIN = BOARD.get_pin('d:11:p')     # azimuth PWM control pin
THETA_MIN = 0.0                         # min. azimuth angle (deg) TODO use
THETA_MAX = 105.0                       # max. azimuth angle (deg) TODO use
THETA_PWM_MIN = 0.40                    # min. azimuth PWM duty cycle
THETA_PWM_MAX = 0.99                    # max. azimuth PWM duty cycle
THETA_HOME = 0.5                        # default azimuth PWM duty cycle

# Elevation servo control
PHI_PIN = BOARD.get_pin('d:10:p')       # elevation PWM control pin
PHI_MIN = 0.0                           # min. elevation angle (deg) TODO use
PHI_MAX = 105.0                         # max. elevation angle (deg) TODO use
PHI_PWM_MIN = 0.40                      # min. elevation PWM duty cycle
PHI_PWM_MAX = 0.99                      # max. elevation PWM duty cycle
PHI_HOME = 0.5                          # default elevation PWM duty cycle

# Camera input
CAM_ID = 1                              # camera ID number (normally 0)
INVERT_X = False                        # invert camera image horizontally
INVERT_Y = True                         # invert camera image vertically
CAM = cv2.VideoCapture(CAM_ID)          # object representing camera
CAM_RES = (160, 120)                    # choose camera resolution
CAM_CENTRE = (int(CAM_RES[0]/2.),       # camera centre x coordinate
              int(CAM_RES[1]/2.))       # camera centre y coordinate
CAM.set(3, CAM_RES[0])                  # set x resolution
CAM.set(4, CAM_RES[1])                  # set y resolution

# Target extraction
INTENSITY_THRESHOLD = 240               # min. pixel intensity for target detection
FLOW_GAUSSIAN = 5
MOVEMENT_THRESHOLD = 7                  # min. optical flow for target detection
NOISE_THRESHOLD = 100000                # min. number of ``target'' px for attack
TARGET_HSV_CODE = np.array([152.64,     # target hue
                            166.055,    # target saturation
                            68.7675])   # target value

# Error minimization (PID loop)
THETA_K_P = 0.001                       # azimuth proportional feedback gain
THETA_K_I = 0.0                         # azimuth integral feedback gain
THETA_K_D = 0.0                         # azimuth derivative feedback gain
PHI_K_P = 0.001                         # elevation proportional feedback gain
PHI_K_I = 0.0                           # elevation integral feedback gain
PHI_K_D = 0.0                           # elevation derivative feedback gain

# Trigger control
GUN0_PIN = BOARD.get_pin('d:8:o')       # gun 0 digital control pin
GUN1_PIN = BOARD.get_pin('d:12:o')      # gun 1 digital control pin
MIN_FIRE_LENGTH = 0.5                   # minimum firing time (to prevent jams) (s)
GUN_DELAY = 0./6.                       # delay between gun 0 and gun 1 firing (s)
BURST_MODE = True                       # enable burst fire mode
REPEAT_BURST_MODE = False               # enable repeat burst fire mode
BURST_LENGTH = 1.0                      # length of a burst (s)
INTER_BURST_INTERVAL = 0.2              # burst cooldown time (repeat burst mode only) (s)
FIRING_RADIUS = CAM_RES[1]/3.           # radius within which to fire if target detected (px)

# Graphical user interface
GUI_RES = (640, 480)                    # graphical user interface resolution
GUI_CENTRE = (int(GUI_RES[0]/2.),       # camera centre x coordinate
              int(GUI_RES[1]/2.))       # camera centre y coordinate

GUI_SCALING = (GUI_RES[0]/float(CAM_RES[0]),
               GUI_RES[1]/float(CAM_RES[1]))
GUI_FIRING_RADIUS = GUI_SCALING[0]*FIRING_RADIUS


"""
Utility functions.
"""

def vprint(msg, lvl=1) :
    """
    Verbose print function -- prints to user only if lvl is less than or equal
    to the global parameter VERBOSITY.
    """
    global VERBOSITY

    if lvl <= VERBOSITY : 
        print(msg)


def run_async(func) :
    """
    Function decorator -- runs func in a separate thread (asynchronously).
    Returns the Thread object handling execution of func.
    """
    @functools.wraps(func)
    def async_func(*args, **kwargs):
	func_hl = threading.Thread(target=func, args=args, kwargs=kwargs)
	func_hl.start()
	return func_hl

    return async_func


def set_cam_resolution(res) :
    """
    Update camera resolution and camera centre coordinates. res should be a
    2-tuple of ints.
    """
    global CAM, CAM_RES, CAM_CENTRE
    CAM_RES = res
    CAM_CENTRE = (int(CAM_RES[0]/2.), int(CAM_RES[1]/2.))
    CAM.set(3, CAM_RES[0])
    CAM.set(4, CAM_RES[1])


"""
Image processing functions.
"""

def grab_frame() :
    """
    Return current camera pixel values in BGR format.
    """
    global INVERT_X, INVERT_Y

    frame = CAM.read()[1]

    if INVERT_X :
        frame = cv2.flip(frame, flipCode=1)

    if INVERT_Y :
        frame = cv2.flip(frame, flipCode=0)

    return frame


def target_from_mask(mask) :
    """
    Compute centroid of a mask, and return [x, y] coordinate pair. If centroid
    cannot be computer, return [None, None].
    """
    M0 = cv2.moments(mask)
    try :
        cx = int(M0['m10']/M0['m00'])
        cy = int(M0['m01']/M0['m00'])
    except ZeroDivisionError :
        cx = None
        cy = None

    if cx is not None : 
        cx_rel = cx - CAM_CENTRE[0]
    else :
        cx_rel = None
    if cy is not None : 
        cy_rel = cy - CAM_CENTRE[1]
    else :
        cy_rel = None

    return [cx_rel, cy_rel]


def extract_target(frame) :
    """
    Extract target's x, y coordinates from a single camera frame. Returns x, y
    coordinates as a 2-list of pixel values and a processed frame illustrating
    the target extraction procedure. If no target was found, x, y coordinates
    will be (None, None).
    """
    #target_xy, proc_frame = color_detect(frame)
    #return target_xy, proc_frame

    target_xy, proc_frame = motion_detect(frame)
    return target_xy, proc_frame

    # convert to grayscale, threshold, open (remove noise), sum
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2)))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3)))

    return target_from_mask(thresh), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def color_detect(frame) :
    """
    Alternative implementation of extract_target() using a color detection
    algorithm rather than simple intensity thresholding. 
    """
    global TARGET_HSV_CODE
    # TODO use opencv's inRange() function

    # apply gaussian blur (denoise), then convert to HSV color space
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

    # calculate difference from target color in hue, then threshold
    H_dist = frame_HSV[:, :, 0] - TARGET_HSV_CODE[0]
    H_dist = 255.*H_dist/float(np.max(H_dist))
    H_dist_thresh = (H_dist > 40).astype(np.uint8)

    # calculate difference from target color in saturation, then threshold
    S_dist = frame_HSV[:, :, 1] - TARGET_HSV_CODE[1]
    S_dist = 255.*S_dist/float(np.max(S_dist))
    S_dist_thresh = (S_dist > 40).astype(np.uint8)

    # find regions where both hue distance and saturation distance are below threshold, then open (remove noise)
    target_mask = 255*H_dist_thresh*S_dist_thresh
    target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, np.ones((2, 2)))

    return target_from_mask(target_mask), cv2.cvtColor(target_mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)



PREV_FRAME = grab_frame()
PREV_GRAY = cv2.cvtColor(PREV_FRAME, cv2.COLOR_BGR2GRAY)
HSV = np.zeros_like(PREV_FRAME)
HSV[..., 1] = 255
def motion_detect(frame) :
    global PREV_FRAME, PREV_GRAY, HSV, CAM_RES

    # convert to grayscale
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(PREV_GRAY, curr_gray, 0.5, 1,
                                        FLOW_GAUSSIAN, 1, 3, 5, 1)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    HSV[...,0] = ang*180/np.pi/2
    HSV[...,2] = mag*10
    #rgb = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    #cv2.imshow('flow', rgb)
    
    # threshold and find centroid
    thresh = cv2.threshold(mag.astype(np.uint8), MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    # store latest frame as new previous frame
    PREV_FRAME = frame
    PREV_GRAY = curr_gray

    return target_from_mask(thresh), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR).astype(np.uint8)


"""
Motor control classes and functions.
"""


def kill_motors() :
    global THETA_HOME, PHI_HOME, THETA_PIN, PHI_PIN
    set_servo_position([THETA_HOME, PHI_HOME])
    time.sleep(0.5)
    THETA_PIN.write(0)
    PHI_PIN.write(0)


class PIDController(object) :
    """
    Minimal implementation of the proportion-integral-derivative (PID) controller.
    """

    def __init__(self, K_p, K_i=0, K_d=0, initial_error=0, max_integral=2000) :
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.max_integral = max_integral

        self.prev_e = initial_error
        self.e_past = max(initial_error, max_integral)


    def update(self, e) :
        self.e_past = max(e + self.e_past, self.max_integral)
        u = self.K_p*e + self.K_i*self.e_past + self.K_d*(e - self.e_prev)
        self.e_prev = e
        return u


CURRENT_ANGLE = [0, 0]
def turning_decision(target_xy_history, xy_prediction_history) :
    """
    Decide how to set servo angles given history of target positions and
    history of target predictions. Returns an angle vector (currently a
    2-element list of PWM duty cycles) and a prediction of the time needed to
    realise this angle vector.
    """
    global CURRENT_ANGLE, THETA_K_P, THETA_K_D, PHI_K_P, PHI_K_D, THETA_HOME, PHI_HOME

    error_x = target_xy_history[-1][0]
    error_y = target_xy_history[-1][1]

    QG = -0.0000005

    if error_x is not None and error_y is not None :
        theta = CURRENT_ANGLE[0] + THETA_K_P*error_x + QG*np.sign(error_x)*error_x**2
        theta = max(min(theta, 1), 0)
        phi = CURRENT_ANGLE[1] + PHI_K_P*error_y + QG*np.sign(error_x)*error_x**2
        phi = max(min(phi, 1), 0)
    else :
        theta = CURRENT_ANGLE[0]
        phi = CURRENT_ANGLE[1]

    latency = 0.0
    CURRENT_ANGLE = [theta, phi]

    return [theta, phi], latency


def set_servo_position(angle_vec) :
    """
    Communicate a pair of servo angles to the arduino. Currently, servo angles
    should be given as a 2-element list of PWM duty cycles.
    """
    global THETA_PIN, THETA_PWM_MAX, THETA_PWM_MIN
    global PHI_PIN, PHI_PWM_MAX, PHI_PWM_MIN

    theta = (THETA_PWM_MAX - THETA_PWM_MIN)*angle_vec[0] + THETA_PWM_MIN
    phi = (PHI_PWM_MAX - PHI_PWM_MIN)*angle_vec[1] + PHI_PWM_MIN
    THETA_PIN.write(theta)
    PHI_PIN.write(phi)


@run_async
def motion_test() :
    while True :
        for angle in np.linspace(0, 1, 1000) :
            set_servo_position([angle, 0])
            time.sleep(0.01)
        for angle in np.linspace(0, 1, 1000)[::-1] :
            set_servo_position([angle, 0])
            time.sleep(0.01)


"""
Fire control functions.
"""

CURRENTLY_FIRING = False
def firing_decision(target_xy_history, xy_prediction_history) :
    """
    Decide whether or not to fire based on history of target positions and
    target predictions. Returns a boolean trigger value -- True => should
    shoot, False => should not shoot.
    """
    global CURRENTLY_FIRING

    target_xy = target_xy_history[-1]

    if target_xy[0] is not None and target_xy[1] is not None :
        if np.linalg.norm(target_xy) <= FIRING_RADIUS and CURRENTLY_FIRING == False :
            vprint('FIRE!!!')
            CURRENTLY_FIRING = True
            return True

        elif np.linalg.norm(target_xy) > FIRING_RADIUS and CURRENTLY_FIRING == True :
            vprint('acquiring target...')
            CURRENTLY_FIRING = False
            return False
    else :
        vprint('acquiring target...')
        CURRENTLY_FIRING = False
        return False


LAST_FIRE = time.time()
@run_async
def fire_guns(trig) :
    """
    Execute firing pattern via arduino, given a boolean trigger value (True =>
    should shoot, False => should not shoot).
    """
    global GUN0_PIN, GUN1_PIN, GUN_PERIOD, CURRENTLY_FIRING, BURST_MODE
    global REPEAT_BURST_MODE, INTER_BURST_INTERVAL, MIN_FIRE_LENGTH, LAST_FIRE

    if time.time() - LAST_FIRE >= MIN_FIRE_LENGTH :
        if trig == True :
            GUN0_PIN.write(1)
            time.sleep(GUN_DELAY)
            GUN1_PIN.write(1)
    
            if BURST_MODE == True :
                time.sleep(BURST_LENGTH)
                GUN0_PIN.write(0)
                GUN1_PIN.write(0)
    
            if REPEAT_BURST_MODE == True :
                time.sleep(BURST_LENGTH)
                GUN0_PIN.write(0)
                GUN1_PIN.write(0)
                time.sleep(INTER_BURST_INTERVAL)
                CURRENTLY_FIRING = False

            LAST_FIRE = time.time()
    
        elif trig == False :
            GUN0_PIN.write(0)
            GUN1_PIN.write(0)


"""
GUI functions.
"""

def draw_crosshair(frame, centre=GUI_CENTRE, radius=GUI_FIRING_RADIUS, color=(0, 0, 0), thickness=1) :
    """
    Draw a crosshair on frame.
    """
    left = int(centre[0] + 1.1*radius)
    right = int(centre[0] - 1.1*radius)
    up = int(centre[1] + 1.1*radius)
    down = int(centre[1] - 1.1*radius)
    cv2.circle(frame, centre, int(radius), color, thickness=thickness)
    cv2.line(frame, (left, centre[1]), (right, centre[1]), color, thickness, 4)
    cv2.line(frame, (centre[0], up), (centre[0], down), color, thickness, 4)


def draw_target(frame, xy, color=(0, 0, 255)) :
    """
    Draw a target indicator on frame.
    """
    cv2.circle(frame, xy, 5, color, thickness=-1)


def gui_frame(frame, proc_frame, target_xy_history, xy_prediction_history, angle_vec, latency, trig) :
    """
    Produce a pretty frame reporting turret state, target state, and target
    prediction to user.
    """
    # TODO there are too many calculations going on here!
    #       precompute the following :
    #           GUI_CENTRE (and use as default for drawing functions)
    #           GUI_SCALING
    #           GUI_FIRING_RADIUS (and use as default for draw_crosshair)
    global CURRENTLY_FIRING, GUI_SCALING
    
    # scale frame and processed frame to GUI dimensions
    frame = cv2.resize(frame, dsize=GUI_RES, interpolation=cv2.INTER_NEAREST)
    proc_frame = cv2.resize(proc_frame, dsize=GUI_RES, interpolation=cv2.INTER_NEAREST)

    # discard non-red channels from processed frame, then overlay onto frame
    proc_frame[:, :, 0] = 0; proc_frame[:, :, 1] = 0
    gui = cv2.addWeighted(frame, 1.0, proc_frame, 0.8, 0, -1)

    if target_xy_history[-1][0] is not None and target_xy_history[-1][1] is not None :
        cx = int(GUI_SCALING[0]*(CAM_CENTRE[0] + target_xy_history[-1][0]))
        cy = int(GUI_SCALING[1]*(CAM_CENTRE[1] + target_xy_history[-1][1]))

        if CURRENTLY_FIRING :
            #cv2.circle(gui, (cx, cy), 5, (0, 0, 255), thickness=-1)
            draw_target(gui, (cx, cy), color=(0, 0, 255))
            draw_crosshair(gui, color=(0, 0, 255), thickness=2)
        else :
            #cv2.circle(gui, (cx, cy), 5, (0, 0, 255), thickness=-1)
            draw_target(gui, (cx, cy), color=(0, 0, 255))
            draw_crosshair(gui, color=(0, 0, 0), thickness=2)
    else :
        draw_crosshair(gui, color=(0, 255, 0), thickness=2)

    return gui


"""
Application functions.
"""

target_xy_history = [[0, 0], [0, 0]]
def main_loop() :
    """
    Execute a selection of the above functions indefinitely.
    """
    while True :
        # grab current frame, extract target x, y position, and store
        frame = grab_frame()
        target_xy, proc_frame = extract_target(frame)
        target_xy_history.append(target_xy)

        # produce a motor command and execute it
        angle_vec, latency = turning_decision(target_xy_history, [])
        if MOVE_ENABLED : set_servo_position(angle_vec)

        # produce a trigger command and execute it
        trig = firing_decision(target_xy_history, [])
        if FIRE_ENABLED : fire_guns(trig)

        # update graphical user interface
        if GUI_ENABLED : 
            gui = gui_frame(frame, proc_frame, target_xy_history, [],
                    angle_vec, latency, trig)
            cv2.imshow('TURRETCAM', gui)

        cv2.waitKey(30)


def terminate() :
    """
    Handle process termination.
    """
    CAM.release()
    cv2.destroyAllWindows()


if __name__ == '__main__' :
    set_servo_position([THETA_HOME, PHI_HOME])
    main_loop()
