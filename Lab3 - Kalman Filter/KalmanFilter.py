import math
import numpy as np
import sympy
from sympy.abc import x, y, theta
from sympy import symbols, Matrix
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from shapely.geometry import LineString
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.stats import plot_covariance


# robot specs, in mm
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100

# rad/s
MOTOR_MAX = 6.3

# map, in mm
MAP_WIDTH = 750
MAP_HEIGHT = 500

# system specs
LASER_STD = 0.03*50  # assumption --> std of 3% of 50mm
IMU_STD = 1  # degree
MOTOR_STD = MOTOR_MAX*0.05  # 5% of motor max


def add_angles(theta1, theta2):
    return (theta1 + theta2) % 360


class EKFRobot(EKF):
    def __init__(self, initial_state):
        EKF.__init__(self, 3, 3, 2)
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_radius = WHEEL_RADIUS

        # x, y, theta
        self.x = np.array([initial_state[0],
                           initial_state[1],
                           initial_state[2]]).T

        x, y, theta, dt, wr, wl = symbols(
            'x, y, theta, dt, wr, wl')

        # Motion Model
        """
            x = x + R(wr + wl)/2 * sin(theta) * dt
            y = y + R(wr + wl)/2 * cos(theta) * dt
            theta = theta + (R/W) * (wr - wl) * dt
        """
        f_xu = Matrix([[x + sympy.sin(theta) * (self.wheel_radius * (wr + wl) / 2) * dt],
                       [y + sympy.cos(theta) * (self.wheel_radius * (wr + wl) / 2) * dt],
                       [theta + (self.wheel_radius / self.width) * (wr - wl) * dt]])

        self.Fj = f_xu.jacobian(Matrix([x, y, theta]))
        self.Wj = f_xu.jacobian(Matrix([wr, wl]))

        self.subs = {x: 0, y: 0, theta: 0, dt: 1, wr: 0, wl: 0}
        self.x_x, self.y_y, self.theta, self.time, self.wr, self.wl = x, y, theta, dt, wr, wl

        self.map_borders = self._init_map_borders()

        self.plot_track = []

    def _drive(self, x, u, dt=1):
        left_vel, right_vel = u
        # maybe???
        left_vel += np.random.randn() * MOTOR_STD
        right_vel += np.random.randn() * MOTOR_STD

        central_vel = self.wheel_radius * (left_vel + right_vel) / 2
        dtheta = -(self.wheel_radius / self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(math.radians(x[2])) * dt
        dy = central_vel * np.cos(math.radians(x[2])) * dt

        du = np.array([dx, dy, math.degrees(dtheta)])
        return x + du

    def _drive_predict(self, x, u, dt=1):
        left_vel, right_vel = u

        central_vel = self.wheel_radius * (left_vel + right_vel) / 2
        dtheta = -(self.wheel_radius / self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(math.radians(self.x[2])) * dt
        dy = central_vel * np.cos(math.radians(self.x[2])) * dt

        du = np.array([dx, dy, math.degrees(dtheta)])
        return x + du

    def _init_map_borders(self):
        west_border = LineString([(0,0), (0, MAP_HEIGHT)])
        east_border = LineString([(MAP_WIDTH,0), (MAP_WIDTH, MAP_HEIGHT)])
        north_border = LineString([(0, MAP_HEIGHT), (MAP_WIDTH, MAP_HEIGHT)])
        south_border = LineString([(0, 0), (MAP_WIDTH, 0)])
        return [west_border, east_border, north_border, south_border]

    def _jh_jacobian_check(self):
        fx, fy, rx, ry = symbols('fx, fy, rx, ry')
        z = Matrix([[sympy.sqrt((fx - x) ** 2 + (fy - y) ** 2)],
                    [sympy.sqrt((rx - x) ** 2 + (ry - y) ** 2)],
                    [sympy.atan2(fy - y, fx - x) - theta]])
        print(z.jacobian(Matrix([x, y, theta])))

    def JH(self, state, landmarks):
        """
        Compute Jacobian of Sensor Readings
        """
        x, y, _ = state
        f_x, f_y, r_x, r_y = landmarks

        f_hyp = (f_x - x)**2 + (f_y - y)**2
        r_hyp = (r_x - x)**2 + (r_y - y)**2
        f_dist = np.sqrt(f_hyp)
        r_dist = np.sqrt(r_hyp)

        JH = np.array([[(-f_x + x)/f_dist, (-f_y + y)/f_dist, 0],
                       [(-r_x + x)/r_dist, (-r_y + y)/r_dist, 0],
                       [-(-f_y + y)/f_hyp, -(f_x - x)/f_hyp, -1]])
        return JH.astype(float)

    def Hx(self, state, landmarks):
        """
        Convert current state to sensor readings.
        """
        x, y, theta = state
        f_x, f_y, r_x, r_y = landmarks

        f_dist = np.sqrt((f_x - x)**2 + (f_y - y)**2)
        r_dist = np.sqrt((r_x - x)**2 + (r_y - y)**2)

        Hx = np.array([f_dist,
                       r_dist,
                       theta])
        return Hx

    def residual(self, z, hx):
        """
        Compute residual: y = z-h(x)
        """
        y = z - hx
        y[2] = y[2] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[2] > np.pi:  # move to [-pi, pi)
            y[2] -= 2 * np.pi
        return y

    def sensor_read(self, state, landmarks):
        """
        Simulate noisy sensor readings.
        """
        x, y, theta = state
        f_x, f_y, r_x, r_y = landmarks

        f_dist = np.sqrt((f_x - x)**2 + (f_y - y)**2)
        r_dist = np.sqrt((r_x - x)**2 + (r_y - y)**2)
        z = np.array([f_dist + np.random.randn() * LASER_STD,
                      r_dist + np.random.randn() * LASER_STD,
                      theta + np.random.randn() * IMU_STD])
        return z

    def _line_calc(self, theta, args=False):
        """
        Helper function for _reflect
        """
        x, y, _ = self.x
        if args:
            x, y = args

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
        Find reflection points of laser range finders.
        """
        x, y, _ = self.x
        f_x, f_y = self._line_calc(theta)
        r_x, r_y = self._line_calc(add_angles(theta, 90))
        front_laser = LineString([(x, y), (f_x, f_y)])
        right_laser = LineString([(x, y), (r_x, r_y)])

        # fx, fy, rx, ry
        landmarks = []
        for i, laser in enumerate([front_laser, right_laser]):
            for border in self.map_borders:
                reflection_point = laser.intersection(border)
                if not reflection_point.is_empty:
                    landmarks.extend([reflection_point.x, reflection_point.y])
                    break

        return np.array(landmarks).astype(float)

    def predict(self, u, dt=1):
        """
        Update localization using control inputs.
        """
        self.x = self._drive_predict(self.x, u, dt)

        self.subs[self.theta] = self.x[2]
        self.subs[self.wl] = u[0]
        self.subs[self.wr] = u[1]

        F = np.array(self.Fj.evalf(subs=self.subs)).astype(float)
        W = np.array(self.Wj.evalf(subs=self.subs)).astype(float)

        # Covariance of Motors
        R = np.array([[MOTOR_STD ** 2, 0],
                      [0, MOTOR_STD ** 2]])

        self.P = np.dot(F, self.P).dot(F.T) + np.dot(W, R).dot(W.T)

    def ekf_update(self, z, landmarks):
        """
        Update localization using observation information.
        """
        self.update(z, HJacobian=self.JH, Hx=self.Hx, residual=self.residual,
                    args=landmarks, hx_args=landmarks)

    def run_localization(self, initial_state, u, dt=1):
        # x, y, theta = initial_state
        # self.x = np.array([x, y, theta]).T
        self.x = np.array([100., 100., 45.]).T
        self.P = np.diag([.1, .1, .1]) # initial covariance
        self.R = np.diag([LASER_STD, LASER_STD, IMU_STD])**2

        plt.figure()
        track = []

        actual_pos = self.x.copy()

        for i in range(10):
            assert self.x.shape == (3,)

            # sim_pos = self._drive_predict(sim_pos, u[i], dt)
            actual_pos = self._drive(actual_pos, u[i], dt)
            self.predict(u[i])
            track.append(self.x)
            self.plot_track.append(actual_pos)

            plot_covariance((self.x[0], self.x[1]), self.P[0:2, 0:2],
                            std=6, facecolor='k', alpha=0.3)

            landmarks = self._reflect(actual_pos[2])
            z = self.sensor_read(actual_pos, landmarks)
            self.ekf_update(z, landmarks)

            plot_covariance((self.x[0], self.x[1]), self.P[0:2, 0:2],
                            std=6, facecolor='g', alpha=0.8)

        track = np.array(track)
        self.plot_track = np.array(self.plot_track)
        plt.plot(track[:, 0], track[:, 1], color='black', lw=2)
        plt.grid()
        plt.title("EKF Robot localization")
        plt.plot(self.plot_track[:, 0], self.plot_track[:, 1], color='r', lw=2)

    def plot_env(self):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.grid()
        plt.xlim((0, MAP_WIDTH))
        plt.ylim((0, MAP_HEIGHT))

        ts = ax.transData
        for state in self.plot_track:
            x, y, theta = state

            frame = plt.Rectangle((x - 45, y - 75), self.width, self.length,
                                  facecolor='cyan', linewidth=1, edgecolor='magenta')
            tr = Affine2D().rotate_deg_around(x, y, -theta)
            t = tr + ts
            frame.set_transform(t)
            ax.add_patch(frame)
            ax.plot(x, y, 'bo', markersize=5)

            f_x, f_y = self._line_calc(theta, (x, y))
            r_x, r_y = self._line_calc(add_angles(theta, 90), (x, y))
            ax.plot([x, f_x], [y, f_y], 'r--')
            ax.plot([x, r_x], [y, r_y], 'r--')


def main():
    nonlinear_traj = True
    initial_state = (200, 300., 90)

    u = []
    if nonlinear_traj:
        for i in range(10):
            if i < 5:
                u.append([1.5, 0.2])
            else:
                u.append([0.8, 4.2])
    else:
        for i in range(10):
            u.append([2, 2])

    robot = EKFRobot(initial_state)
    robot.run_localization(initial_state, u)
    robot.plot_env()
    plt.show()


if __name__ == "__main__":
    main()