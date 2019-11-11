import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from shapely.geometry import LineString
import utils


# in mm
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100

# map
MAP_WIDTH = 750
MAP_HEIGHT = 500

# system specs
LASER_STD = 1200 * 0.03
IMU_STD = np.radians(0.1)
MOTOR_STD = 60 * 0.05


class Robot:
    def __init__(self, initial_state):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_radius = WHEEL_RADIUS

        # x, y, theta
        self.state = np.array([initial_state[0], initial_state[1], initial_state[2]])

        self.map_borders = self._init_map()

    def _init_map(self):
        west_border = LineString([(0,0), (0, MAP_HEIGHT)])
        east_border = LineString([(MAP_WIDTH,0), (MAP_WIDTH, MAP_HEIGHT)])
        north_border = LineString([(0, MAP_HEIGHT), (MAP_WIDTH, MAP_HEIGHT)])
        south_border = LineString([(0,0), (MAP_WIDTH, 0)])
        return [west_border, east_border, north_border, south_border]


    def _drive(self, u, dt=0.1):
        left_vel, right_vel = u
        central_vel = self.wheel_radius * (left_vel + right_vel) / 2
        dtheta = -(self.wheel_radius / self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(math.radians(self.state[2])) * dt
        dy = central_vel * np.cos(math.radians(self.state[2])) * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += math.degrees(dtheta)

    def _line_calc(self, theta):
        x, y, _ = self.state

        # Special Cases
        if theta == 0:
            return x, MAP_HEIGHT
        elif theta == 180:
            return x, 0
        elif theta == 90:
            return MAP_WIDTH, y
        elif theta == 270:
            return 0, y

        slope = 1.0 / np.tan(math.radians(theta))
        if theta < 180:
            f_x = 800  # arbitrary value past map border
            f_y = slope * f_x + (-slope * x + y)
        else:
            f_x = -100  # arbitrary value past map border
            f_y = slope * f_x + (-slope * x + y)

        return f_x, f_y

    def _reflect(self, theta):
        """
        Use two laser range sensors - VL53L0X
        :return:
        """
        x, y, _ = self.state
        r_x, r_y = self._line_calc(theta)
        laser = LineString([(x, y), (r_x, r_y)])
        for border in self.map_borders:
            reflection_point = laser.intersection(border)
            if not reflection_point.is_empty:
                break

        return r_x, r_y

    def Hx(self, state):
        x, y, theta = state
        front_x, front_y = self._reflect(theta)
        right_x, right_y = self._reflect(utils.add_angles(theta, 90))

        front_laser = np.sqrt((front_x - x)**2 + (front_y - y)**2)
        right_laser = np.sqrt((right_x - x)**2 + (right_y - y)**2)

        return np.array([front_laser, right_laser, theta])


    def plot_env(self):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.grid()
        plt.xlim((0, MAP_WIDTH))
        plt.ylim((0, MAP_HEIGHT))

        ts = ax.transData
        x, y, theta = self.state
        frame = plt.Rectangle((x-45, y-75), self.width, self.length,
                              facecolor='cyan', linewidth=1, edgecolor='magenta')
        tr = Affine2D().rotate_deg_around(x, y, -theta)
        t = tr + ts
        frame.set_transform(t)
        ax.add_patch(frame)
        ax.plot(x, y, 'bo', markersize=5)

        f_x, f_y = self._line_calc(theta)
        r_x, r_y = self._line_calc(utils.add_angles(theta, 90))
        ax.plot([x, f_x], [y, f_y], 'r--')
        ax.plot([x, r_x], [y, r_y], 'r--')

    def IMU(self):
        """
        Inertial measurement unit - MPU-9250
        :return:
        """
        return self.state[2]


    # def initialize_system(self):
    #
    #     A =


def main():
    initial_state = (375, 250, 15)
    robot = Robot(initial_state)
    robot.plot_env()
    plt.show()






if __name__ == "__main__":
    main()