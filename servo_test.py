# /usr/bin/env python

import time
import numpy as np
from pyfirmata import Arduino


"""
Constant declarations.
"""

ARDUINO_PORT = '/dev/ttyACM0'
print('Connecting to arduino at ' + str(ARDUINO_PORT) + '...')
BOARD = Arduino(ARDUINO_PORT)
print('Connected OK!')


THETA_PIN = BOARD.get_pin('d:11:p')
THETA_MIN = 0.0
THETA_MAX = 105.0
THETA_PWM_MIN = 0.40
THETA_PWM_MAX = 0.99

PHI_PIN = BOARD.get_pin('d:10:p')
PHI_MIN = 0.0
PHI_MAX = 105.0
PHI_PWM_MIN = 0.40
PHI_PWM_MAX = 0.99

GUN0_PIN = BOARD.get_pin('d:13:o')
GUN1_PIN = BOARD.get_pin('d:12:o')


def position_command(mvec) :
    global THETA_PWM_MAX, THETA_PWM_MIN, THETA_PIN
    global PHI_PWM_MAX, PHI_PWM_MIN, PHI_PIN

    theta = (THETA_PWM_MAX - THETA_PWM_MIN)*mvec[0] + THETA_PWM_MIN
    phi = (PHI_PWM_MAX - PHI_PWM_MIN)*mvec[1] + PHI_PWM_MIN
    THETA_PIN.write(theta)
    PHI_PIN.write(phi)


def kill_motors() :
    global THETA_PIN, PHI_PIN
    THETA_PIN.write(0)
    PHI_PIN.write(0)


print('Homing servomotors on ' + str(THETA_PIN) + ' and ' + str(PHI_PIN))
position_command([0, 0])
time.sleep(0.5)
print('Servomotors OK!')


def test_positioning(n=1000, delay = 0.001) :
    thetas = np.linspace(0, 1, n)
    phis = np.linspace(0, 1, n)

    def _run(thetas, phis) :
        for theta, phi in zip(thetas, phis) :
            position_command([theta, phi])
            time.sleep(delay)
        
    while True :
        _run(thetas, phis)
        _run(thetas[::-1], phis[::-1])
