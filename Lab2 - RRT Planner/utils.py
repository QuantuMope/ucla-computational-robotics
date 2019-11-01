import numpy as np

"""
    Helper functions for RRT Robot
"""


# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100


def rpm_to_vel(rpm):
    """
    Convert from RPM to translational velocity.

    :param rpm: wheel RPM
    :return distance_per_sec: wheel velocity
    """
    if rpm > 60 or rpm < -60:
        raise ValueError("Invalid rpm. Range: -60 to 60")
    tire_circum = 2*np.pi*WHEEL_RADIUS
    rps = rpm/60
    distance_per_sec = (tire_circum * rps)
    return distance_per_sec


def vel_to_rpm(vel):
    """
    Convert from translational velocity to RPM.

    :param vel: wheel velocity
    :return rpm: wheel rpm
    """
    if vel > 157 or vel < -157:
        raise ValueError("Invalid velocity. Range: -157 to 157")
    tire_circum = 2*np.pi*WHEEL_RADIUS
    rps = vel / tire_circum
    rpm = rps * 60
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
