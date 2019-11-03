import numpy as np
import math
import time

"""
    Helper functions for RRT Robot
"""


# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100


def rpm_to_vel(rpm):
    """
    Convert from RPM to angular velocity.

    :param rpm: wheel RPM
    :return angular_vel: wheel angular velocity (rad/sec)
    """
    if rpm > 60 or rpm < -60:
        raise ValueError("Invalid rpm. Range: -60 to 60 RPM")
    angular_vel = math.radians(rpm * 6)
    return angular_vel


def vel_to_rpm(vel):
    """
    Convert from angular velocity to RPM.

    :param vel: wheel angular velocity (rad/sec)
    :return rpm: wheel rpm
    """
    if vel > 6.3 or vel < -6.3:
        raise ValueError("Invalid angular velocity. Range: -6.3 to 6.3 rad/sec")
    rpm = math.degrees(vel) / 6
    return rpm


def add_angles(a, b):
    """
    Helper function to to correctly add angles and keep angle within 0-360 degrees.
    """
    if a+b > 360:
        return a+b-360
    elif a+b < 0:
        return 360+(a+b)
    return a+b


def obstacle_to_corner(obstacle):
    """
    Helper function to convert obstacle to coordinates of its corners
    """
    bottom = obstacle[1]
    top = bottom + obstacle[3]
    left = obstacle[0]
    right = left + obstacle[2]
    return bottom, top, left, right


def sample_random_point(low_x, high_x, low_y, high_y, low_theta, high_theta):
    """
    Sample a random point in the state space.
    :params: lower and upper bounds of each state dimension
    :return: numpy array [x, y, theta] of sampled point
    """
    rand_x = np.random.uniform(low_x, high_x+1)
    rand_y = np.random.uniform(low_y, high_y+1)
    rand_theta = np.random.uniform(low_theta, high_theta)
    return np.array([rand_x, rand_y, rand_theta])

class Timer:
    def __init__(self):
        self._elapsed_time  = 0
        self._startTime      = 0
        self._cycled_count  = 0

    def tic(self):
        self._startTime = time.time()

    def toc(self):
        return time.time() - self._startTime

    def estimated_remaining_time(self,cycle_time,current_count,_max_count):
        self._elapsed_time += cycle_time
        self._cycled_count += 1
        _time_for_one_cycle = self._elapsed_time/self._cycled_count

        return (_max_count - current_count) * _time_for_one_cycle




