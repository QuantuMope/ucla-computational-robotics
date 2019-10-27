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
    return np.round(distance_per_sec)


def vel_to_rpm(vel):
    if vel > 157 or vel < -157:
        raise ValueError("Invalid velocity. Range: -157 to 157")
    tire_circum = 2*np.pi*WHEEL_RADIUS
    rps = vel / tire_circum
    rpm = rps * 60
    return np.round(rpm)


def add_angles(a, b):
    if a+b > 360:
        return np.round(a+b-360)
    elif a+b < 0:
        return np.round(360+(a+b))
    return np.round(a+b)
