import numpy as np

# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100


def rpm_to_vel(rpm):
    if rpm > 60 or rpm < -60:
        raise ValueError("Invalid rpm. Range: -60 to 60")
    tire_circum = 2*np.pi*WHEEL_RADIUS
    rps = rpm/60
    distance_per_sec = (tire_circum * rps)
    return distance_per_sec


def vel_to_rpm(vel):
    if vel > 157 or vel < -157:
        raise ValueError("Invalid velocity. Range: -157 to 157")
    tire_circum = 2*np.pi*WHEEL_RADIUS
    rps = vel / tire_circum
    rpm = rps * 60
    return rpm


def add_angles(a, b):
    if a+b > 360:
        return a+b-360
    elif a+b < 0:
        return 360+(a+b)
    return a+b


def obstacle_to_corner(obstacle):
    bottom = obstacle[1]
    top = bottom + obstacle[3]
    left = obstacle[0]
    right = left + obstacle[2]
    return bottom, top, left, right


def sample_random_point(low_x, high_x, low_y, high_y, low_theta, high_theta):
    rand_x = np.random.uniform(low_x, high_x+1)
    rand_y = np.random.uniform(low_y, high_y+1)
    rand_theta = np.random.uniform(low_theta, high_theta+1)
    return rand_x, rand_y, rand_theta
