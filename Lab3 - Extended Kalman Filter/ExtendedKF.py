import numpy as np
import sympy
from sympy.abc import x, y, theta
from sympy import symbols, Matrix
from math import degrees, radians
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from shapely.geometry import LineString
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.stats import plot_covariance


# robot specs in mm
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100

# max motor speed 60 RPM --> 6.3 rad/s
MOTOR_MAX = 6.3

# map dimensions in mm
MAP_WIDTH = 750
MAP_HEIGHT = 500

# variance specs
LASER_STD = 0.03*50  # assumption --> std of 3% of 50mm
IMU_STD = 1  # degree
MOTOR_STD = MOTOR_MAX*0.05  # 5% of motor max


def add_angles(theta1, theta2):
    return (theta1 + theta2) % 360


class EKFRobot(EKF):
    """
        Two-wheeled non-holonomic differential drive robot.
        Uses Extended Kalman Filter for state estimation.

        Sensors:
            Two laser range sensors - VL53L0X
            noise: std of 3% of 50mm

            One inertial measurment unit - MPU-9250
            noise: std of 1 degree

        Motors:
            Two b-directional continuous rotation servos - FS90R
            Capable of -60 to 60 RPM
            noise: std of 5% of Motor Max
    """
    def __init__(self, initial_state):
        EKF.__init__(self, 3, 3, 2)
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_radius = WHEEL_RADIUS

        # Robot state definition. (x, y, theta)
        # Estimated state of the robot.
        self.x = np.array([initial_state[0],
                           initial_state[1],
                           initial_state[2]]).T

        # Sympy symbols for Jacobian computation.
        x, y, theta, dt, wr, wl = symbols('x, y, theta, dt, wr, wl')

        """
                       Motion Model
        ---------------------------------------------- 
            x = x + R(wr + wl)/2 * sin(theta) * dt
            y = y + R(wr + wl)/2 * cos(theta) * dt
            theta = theta + (R/W) * (wr - wl) * dt
        ----------------------------------------------
        """

        # Motion model in sympy matrix.
        f_xu = Matrix([[x + sympy.sin(theta) * (self.wheel_radius * (wr + wl) / 2) * dt],
                       [y + sympy.cos(theta) * (self.wheel_radius * (wr + wl) / 2) * dt],
                       [theta + (self.wheel_radius / self.width) * (wr - wl) * dt]])

        # Jacobians of motion model w.r. states and control inputs.
        self.Fj = f_xu.jacobian(Matrix([x, y, theta]))
        self.Wj = f_xu.jacobian(Matrix([wr, wl]))

        # Initialize variable substitutions.
        self.subs = {x: 0, y: 0, theta: 0, dt: 1, wr: 0, wl: 0}
        # Allow for dictionary indexing in later functions.
        self.x_x, self.y_y, self.theta, self.time, self.wr, self.wl = x, y, theta, dt, wr, wl

        self.map_borders = self._init_map_borders()

        self.esti_states = []
        self.true_states = []

    def _drive_real(self, x_true, u, dt=1):
        """
        Simulate driving with control inputs for true states.
        Generate realistic control noise according to motor covariance.

        :param x_true: The true current state of the robot. (x, y, theta)_true
        :param u: The control inputs. (wl, wr)
        :param dt: Time in between control inputs.
        :return: New true state of robot.
        """
        wl, wr = u

        # Add zero mean Gaussian noise.
        wl += np.random.randn() * MOTOR_STD
        wr += np.random.randn() * MOTOR_STD

        central_vel = self.wheel_radius * (wl + wr) / 2
        dtheta = -(self.wheel_radius / self.width) * (wr - wl) * dt
        dx = central_vel * np.sin(radians(x_true[2])) * dt
        dy = central_vel * np.cos(radians(x_true[2])) * dt

        du = np.array([dx, dy, degrees(dtheta)])
        return x_true + du

    def _drive_predict(self, x_predicted, u, dt=1):
        """
        Simulate driving with control inputs for predicted states.
        Predicting with no control noise.

        :param x_predicted: The current predicted state of the robot. (x, y, theta)_predicted
        :param u: The control inputs. (wl, wr)
        :param dt: Time in between control inputs.
        :return: New predicted state of robot.
        """
        wl, wr = u

        central_vel = self.wheel_radius * (wl + wr) / 2
        dtheta = -(self.wheel_radius / self.width) * (wr - wl) * dt
        dx = central_vel * np.sin(radians(x_predicted[2])) * dt
        dy = central_vel * np.cos(radians(x_predicted[2])) * dt

        du = np.array([dx, dy, degrees(dtheta)])
        return x_predicted + du

    def _init_map_borders(self):
        """
        Initialize the map borders in terms of shapely linestrings.

        :return: array of map borders
        """
        west_border = LineString([(0, 0), (0, MAP_HEIGHT)])
        east_border = LineString([(MAP_WIDTH, 0), (MAP_WIDTH, MAP_HEIGHT)])
        north_border = LineString([(0, MAP_HEIGHT), (MAP_WIDTH, MAP_HEIGHT)])
        south_border = LineString([(0, 0), (MAP_WIDTH, 0)])
        return [west_border, east_border, north_border, south_border]

    def _jh_jacobian_check(self):
        """
        Helper function that prints the symbolic Jacobian of sensor model.

        Used only to aid in coding JH.
        """

        # Sympy symbols for laser range finder lengths in x and y.
        fx, fy, rx, ry = symbols('fx, fy, rx, ry')

        # Sensor model.
        z = Matrix([[sympy.sqrt((fx - x) ** 2 + (fy - y) ** 2)],
                    [sympy.sqrt((rx - x) ** 2 + (ry - y) ** 2)],
                    [sympy.atan2(fy - y, fx - x) - theta]])
        print(z.jacobian(Matrix([x, y, theta])))

    def _JH(self, state, landmarks):
        """
        Computes the Jacobian of the sensor model H(x) w.r. to state.
        :param state: current estimated state
        :param landmarks: laser range finder distances in x and y
        :return JH: Jacobian of H(x) w.r. to x
        """
        x, y, _ = state
        f_x, f_y, r_x, r_y = landmarks

        f_hyp = (f_x - x)**2 + (f_y - y)**2
        r_hyp = (r_x - x)**2 + (r_y - y)**2
        f_dist = np.sqrt(f_hyp)
        r_dist = np.sqrt(r_hyp)

        # Obtained from self._jh_jacobian_check()
        JH = np.array([[(-f_x + x)/f_dist, (-f_y + y)/f_dist, 0],
                       [(-r_x + x)/r_dist, (-r_y + y)/r_dist, 0],
                       [-(-f_y + y)/f_hyp, -(f_x - x)/f_hyp, -1]])
        return JH.astype(float)

    def _Hx(self, state, landmarks):
        """
        Sensor model H(x).
        Converts current state to sensor readings.
        :param state: current true state
        :param landmarks: laser range finder distances in x and y
        :return Hx: sensor outputs [front laser distance,
                                    right laser distance,
                                    angular pose]
        """
        x, y, theta = state
        f_x, f_y, r_x, r_y = landmarks

        # Calculate laser distances.
        f_dist = np.sqrt((f_x - x)**2 + (f_y - y)**2)
        r_dist = np.sqrt((r_x - x)**2 + (r_y - y)**2)

        Hx = np.array([f_dist,
                       r_dist,
                       theta])
        return Hx

    def _residual(self, z, hx):
        """
        Compute the residual of sensors, y = z-h(x)
        Make sure that angles are within range 0-360 deg.

        :param z: sensor readings
        :param hx: predicted sensor readings from sensor model
        :return y: residual
        """
        y = z - hx
        y[2] = y[2] % 360
        return y

    def _sensor_read(self, state, landmarks):
        """
        Simulate noisy sensor readings z = [front laser distance,
                                            right laser distance,
                                            IMU angular pose]
        :param state: current state
        :param landmarks: laser range finder distances in x and y
        """
        x, y, theta = state
        f_x, f_y, r_x, r_y = landmarks

        # Calculate laser distances.
        f_dist = np.sqrt((f_x - x)**2 + (f_y - y)**2)
        r_dist = np.sqrt((r_x - x)**2 + (r_y - y)**2)

        # Simulate sensor noise.
        z = np.array([f_dist + np.random.randn() * LASER_STD,
                      r_dist + np.random.randn() * LASER_STD,
                      theta + np.random.randn() * IMU_STD])
        return z

    def _line_calc(self, theta, args=False):
        """
        Helper function for _reflect.
        Calculates the slope of the laser dependent on the theta
        and then calculates coordinates for the laser line.

        Coordinates are used with shapely's LineString class to
        find the intersection point between the laser and map border.

        :param theta: the angular pose of robot
        :param args: override using estimated state
        """
        # Use current estimated state unless specified otherwise.
        x1, y1, _ = self.x
        if args:
            x1, y1 = args

        # Special Cases that make calculating slope impossible
        if theta == 0:
            return x1, MAP_HEIGHT
        elif theta == 180:
            return x1, 0
        elif theta == 90:
            return MAP_WIDTH, y1
        elif theta == 270:
            return 0, y1

        slope = 1.0 / np.tan(radians(theta))

        # Arbitrary value that lies outside map border
        if theta < 180:
            x2 = 800
        else:
            x2 = -100

        y2 = slope * x2 + (-slope * x1 + y1)

        return x2, y2

    def _reflect(self, theta):
        """
        Helper function for computing landmark points.
        Landmark points are the x and y coordinates of both laser range finders.

        :param theta: robot angular pose
        :return landmarks: return numpy array of laser distance in x and y
        """

        # Calculate the lines regarding the front and right lasers.
        x, y, _ = self.x
        f_x, f_y = self._line_calc(theta)
        r_x, r_y = self._line_calc(add_angles(theta, 90))
        front_laser = LineString([(x, y), (f_x, f_y)])
        right_laser = LineString([(x, y), (r_x, r_y)])

        # Find the intersection between the lasers and map borders.
        landmarks = []  # fx, fy, rx, ry
        for laser in [front_laser, right_laser]:
            for border in self.map_borders:
                reflection_point = laser.intersection(border)
                if not reflection_point.is_empty:
                    landmarks.extend([reflection_point.x, reflection_point.y])
                    break

        return np.array(landmarks).astype(float)

    def _predict(self, u, dt=1):
        """

        :param u: control inputs
        :param dt: length of time control input is done
        :return: updated covariance matrix
        """
        # Predict next state.
        self.x = self._drive_predict(self.x, u, dt)

        # Update Jacobian variable substitutions.
        self.subs[self.theta] = self.x[2]
        self.subs[self.wl] = u[0]
        self.subs[self.wr] = u[1]

        # Calculate Jacobians.
        F = np.array(self.Fj.evalf(subs=self.subs)).astype(float)
        W = np.array(self.Wj.evalf(subs=self.subs)).astype(float)

        # Covariance of Motors
        R = np.array([[MOTOR_STD ** 2, 0],
                      [0, MOTOR_STD ** 2]])

        self.P = np.dot(F, self.P).dot(F.T) + np.dot(W, R).dot(W.T)

    def _ekf_update(self, z, landmarks):
        """
        Update state estimation using observation information.
        Use noisy sensor readings, sensor model jacobian,
        sensor reading predictions, and landmarks to update state.

        Use filterpy library's EKF class to perform matrix operations.
        Reference: https://github.com/rlabbe/filterpy

        :param z: noisy sensor readings
        :param landmarks: front and right laser distance in x and y
        """
        self.update(z, HJacobian=self._JH, Hx=self._Hx, residual=self._residual,
                    args=landmarks, hx_args=landmarks)

    def run_localization(self, initial_state, u, dt=1,
                         increased_resolution=False, unknown_location=False):

        # Initialize
        x, y, theta = initial_state
        self.x = np.array([x, y, theta]).T

        # Initialize state covariance.
        self.P = np.diag([.1, .1, .1])

        # Sensor covariance matrix.
        self.R = np.diag([LASER_STD, LASER_STD, IMU_STD])**2

        # If robot starts with unknown location assign a random state.
        if unknown_location:
            true_state = np.array([np.random.uniform(100, MAP_WIDTH-100),
                                   np.random.uniform(100, MAP_HEIGHT-100),
                                   np.random.uniform(0, 360)]).T
        else:
            true_state = self.x.copy()

        if increased_resolution:
            total_scans = 100
        else:
            total_scans = 10

        counter = 0
        plt.figure()
        plt.grid()
        self.esti_states = [self.x]
        self.true_states.append(true_state)

        for t in range(total_scans):
            assert self.x.shape == (3,)

            if increased_resolution:
                t = t % 10
                dt_n = dt / 10
            else:
                dt_n = dt

            # Update the true state using noisy control inputs.
            true_state = self._drive_real(true_state, u[t], dt_n)

            # Predict estimated state using expected control inputs.
            self._predict(u[t], dt_n)

            # Store estimated and true states to plot later.
            self.esti_states.append(self.x)
            self.true_states.append(true_state)

            # Plot pre-observation belief state. Covariance ellipse.
            if counter % 10 == 0 or not increased_resolution:
                plot_covariance((self.x[0], self.x[1]), self.P[0:2, 0:2],
                                std=6, facecolor='k', alpha=0.3)

            # Get noisy sensor readings based on robot true state.
            landmarks = self._reflect(true_state[2])
            z = self._sensor_read(true_state, landmarks)

            # Update estimated state based on noisy sensor readings.
            self._ekf_update(z, landmarks)

            # Plot post-observation belief state. Covariance ellipse.
            if counter % 10 == 0 or not increased_resolution:
                plot_covariance((self.x[0], self.x[1]), self.P[0:2, 0:2],
                                std=6, facecolor='g', alpha=0.8)
            counter += 1

        # Plot the true state and estimated state trajectories.
        self.esti_states = np.array(self.esti_states)
        self.true_states = np.array(self.true_states)
        plt.plot(self.esti_states[:, 0], self.esti_states[:, 1], color='r', lw=2)
        plt.plot(self.true_states[:, 0], self.true_states[:, 1], color='black', lw=2)
        plt.plot(self.esti_states[:, 0], self.esti_states[:, 1], 'ro')
        plt.plot(self.true_states[:, 0], self.true_states[:, 1], 'bo')

    def plot_env(self):
        """
        Plot the true robot trajectory and estimated trajectory.
        """
        fig, ax = plt.subplots(figsize=(7.5, 5))
        plt.xlim((0, MAP_WIDTH))
        plt.ylim((0, MAP_HEIGHT))

        ts = ax.transData
        for state in self.true_states:
            x, y, theta = state

            # Plot the robot body.
            frame = plt.Rectangle((x - 45, y - 75), self.width, self.length,
                                  facecolor='cyan', linewidth=1, edgecolor='magenta')
            tr = Affine2D().rotate_deg_around(x, y, -theta)
            t = tr + ts
            frame.set_transform(t)
            ax.add_patch(frame)

            # Plot the lasers.
            f_x, f_y = self._line_calc(theta, (x, y))
            r_x, r_y = self._line_calc(add_angles(theta, 90), (x, y))
            ax.plot([x, f_x], [y, f_y], 'r--', lw=1)
            ax.plot([x, r_x], [y, r_y], 'r--', lw=1)

        # Plot the true state and estimated state trajectories.
        plt.plot(self.esti_states[:, 0], self.esti_states[:, 1], color='r', lw=2)
        plt.plot(self.true_states[:, 0], self.true_states[:, 1], color='black', lw=2)
        plt.plot(self.esti_states[:, 0], self.esti_states[:, 1], 'ro')
        plt.plot(self.true_states[:, 0], self.true_states[:, 1], 'bo')


def main():

    # ====================== SIMULATION INTERFACE =========================
    initial_state = (100, 100., 60.)
    nonlinear_traj = False
    increased_resolution = False
    unknown_location = False
    # =====================================================================

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
    robot.run_localization(initial_state, u,
                           increased_resolution=increased_resolution,
                           unknown_location=unknown_location)
    robot.plot_env()
    plt.show()


if __name__ == "__main__":
    main()