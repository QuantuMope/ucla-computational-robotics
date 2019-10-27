import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ParkingLot import ParkingLot
from utils import rpm_to_vel, vel_to_rpm, add_angles


# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100


class RRT_Node:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None
        self.actions = []

    def get_xy(self):
        return np.array(self.x, self.y)

    # def add_node(self, ):



class RRT_Robot:
    def __init__(self, start_x, start_y, start_theta, parking_plot):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_rad = WHEEL_RADIUS

        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        self.ax = parking_plot

        self.frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length,
                                   facecolor='cyan', linewidth=1, edgecolor='magenta')
        self.ax.plot(self.x, self.y, marker='o', color='blue')
        self.ax.add_patch(self.frame)

        self.RRT = RRT_Node(start_x, start_y, start_theta)
        self.node_list = [self.RRT]
        self.config_space = []

    def calc_drive(self, x, y, theta, left, right, dt=0.1):
        left_vel = rpm_to_vel(left)
        right_vel = rpm_to_vel(right)
        central_vel = (left_vel + right_vel) / 2
        dtheta = -(self.wheel_rad/self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(theta*2*np.pi/360) * dt
        dy = central_vel * np.cos(theta*2*np.pi/360) * dt

        theta = add_angles(theta, dtheta)
        x += dx
        y += dy
        frame = plt.Rectangle((x-45, y-75), self.width, self.length)
        ts = self.ax.transData
        tr = Affine2D().rotate_deg_around(x, y, -theta)
        t = tr + ts
        frame.set_transform(t)

        return dx, dy, int(dtheta)

    def drive(self, left, right, dt=0.1):
        left_vel = rpm_to_vel(left)
        right_vel = rpm_to_vel(right)
        central_vel = (left_vel + right_vel) / 2
        dtheta = -(self.wheel_rad/self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(self.theta*2*np.pi/360) * dt
        dy = central_vel * np.cos(self.theta*2*np.pi/360) * dt

        self.theta = add_angles(self.theta, dtheta)
        self.x += dx
        self.y += dy
        self.frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length,
                                   facecolor='cyan', linewidth=1, edgecolor='magenta')
        ts = self.ax.transData
        tr = Affine2D().rotate_deg_around(self.x, self.y, -self.theta)
        t = tr + ts
        self.frame.set_transform(t)
        self.ax.add_patch(self.frame)
        self.ax.plot(self.x, self.y, marker='o', color='blue')

    def calculate_config_space(self, obstacles):
        config_frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length)
        ts = self.ax.transData
        for theta in range(0, 361):
            tr = Affine2D().rotate_deg_around(self.x, self.y, -theta)
            t = tr + ts
            config_frame.set_transform(t)
            x, y = config_frame.get_x(), config_frame.get_y()

            # bottom left, top left, top right, bottom right
            corners = np.array([[x, y],
                                [x, y+self.length],
                                [x+self.width, y+self.length],
                                [x+self.width, y]])
            corners = tr.transform(corners)
            corner_to_center = (corners - np.array([self.x, self.y])) * -1
            obstacle_boundaries = []
            for obstacle in obstacles:
                boundary = []
                bottom = obstacle[1]
                top = bottom + obstacle[3]
                left = obstacle[0]
                right = left + obstacle[2]
                a, b, c, d = 0, 1, 2, 3
                if theta < 180:
                    a, b, c, d = 3, 0, 1, 2
                boundary.append((left + corner_to_center[a][0], top + corner_to_center[a][1], theta))
                boundary.append((right + corner_to_center[a][0], top + corner_to_center[a][1], theta))
                boundary.append((right + corner_to_center[b][0], top + corner_to_center[b][1], theta))
                boundary.append((right + corner_to_center[b][0], bottom + corner_to_center[b][1], theta))
                boundary.append((right + corner_to_center[c][0], bottom + corner_to_center[c][1], theta))
                boundary.append((left + corner_to_center[c][0], bottom + corner_to_center[c][1], theta))
                boundary.append((left + corner_to_center[d][0], bottom + corner_to_center[d][1], theta))
                boundary.append((left + corner_to_center[d][0], top + corner_to_center[d][1], theta))
                obstacle_boundaries.append(boundary)
            self.config_space.append(obstacle_boundaries)

    def plot_config_space(self):
        fig = plt.figure(2)
        ax = Axes3D(fig)
        ax.set_xlim3d((0, 2000))
        ax.set_ylim3d((0, 1400))
        ax.set_zlim3d((0, 360))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Theta")
        plt.title("Configuration Space")
        for theta, all_obstacles in enumerate(self.config_space):
            color = 'steelblue'
            if theta % 20 == 0: color = 'black'
            for obstacle_verts in all_obstacles:
                ax.add_collection3d(Poly3DCollection([obstacle_verts], edgecolors=color))

    # Question 2(a)
    def find_nearest_node(self, xrand):
        shortest_dist = np.linalg.norm(self.RRT.get_xy() - xrand)
        min_index = 0
        nearest_node = self.RRT
        for i, node in enumerate(self.node_list):
            dist = np.linalg.norm(node.get_xy() - xrand)
            if dist < shortest_dist:
                shortest_dist = dist
                min_index = i
                nearest_node = node
        return nearest_node, min_index

    # Question 2(b)
    def generate_trajectory(self, initial_state, target_state, dt=0.1):
        x_start = x_curr = initial_state[0]
        y_start = y_curr = initial_state[1]
        theta_start = theta_curr = initial_state[2]
        x_goal = target_state[0]
        y_goal = target_state[1]
        theta_goal = target_state[2]

        t = 0
        control_inputs = []
        trajectory = [[x_start, y_start, theta_start]]
        self.ax.plot(x_start, y_start, 'ro')
        self.ax.plot(x_goal, y_goal, 'go')

        while t < 1:
            x_diff = x_goal - x_curr
            y_diff = y_goal - y_curr
            theta_diff = theta_goal - theta_curr

            """
            System of Equations
            
            -(self.wheel_rad/self.width) * (right_vel - left_vel) = theta_diff             
            ((left_vel + right_vel)/2) * np.sin(theta_start*2*np.pi/360) = x_diff
            ((left_vel + right_vel)/2) * np.cos(theta_start*2*np.pi/360) = y_diff
            """

            def equations(p):
                left_vel, right_vel = p
                equation1 = -(self.wheel_rad / self.width) * (right_vel - left_vel) - theta_diff
                equation2 = ((left_vel + right_vel) / 2) * np.sin(theta_start * 2 * np.pi / 360) - x_diff
                equation3 = ((left_vel + right_vel) / 2) * np.cos(theta_start * 2 * np.pi / 360) - y_diff
                return np.abs([equation1, equation2, equation3])

            bnds = ([-157, 157])
            res = least_squares(equations, [1, 1], bounds=bnds)
            left_vel_input = res.x[0]
            right_vel_input = res.x[1]
            dx, dy, dtheta = self.calc_drive(x_curr, y_curr, theta_curr, vel_to_rpm(left_vel_input),
                                             vel_to_rpm(right_vel_input), dt)

            x_curr += dx
            y_curr += dy
            theta_curr = add_angles(theta_curr, dtheta)
            
            if self.check_collision((x_curr, y_curr, theta_curr)):
                control_inputs = None
                trajectory = None
                break

            control_inputs.append([vel_to_rpm(left_vel_input), vel_to_rpm(right_vel_input)])
            trajectory.append([x_curr, y_curr, theta_curr])
            t += dt
        return control_inputs, trajectory

    # Question 2(c)
    def check_collision(self, state):
        if len(self.config_space) == 0:
            raise ValueError("Robot configuration space has not yet been computed. \n \
                              Please use calculate_config_space.")
        collision = False
        x, y, theta = state
        point = Point(x, y)
        config_space_plane = self.config_space[theta]
        for obstacle in config_space_plane:
            coordinates2d = [(x, y) for x, y, theta in obstacle]
            polygon = Polygon(coordinates2d)
            collision = polygon.contains(point)
            if collision: break
        return collision

    def sample_xrand(self):


    def plan(self, initial_state, target_state):

            











def main():
    env = ParkingLot()

    robot = RRT_Robot(100, 100, 0, env.ax)
    robot.calculate_config_space(env.obstacles)
    # robot.plot_config_space()
    control_inputs, traj = robot.generate_trajectory((100, 100, 0), (550, 550, 60))
    print(control_inputs)
    plt.show()



if __name__ == "__main__":
    main()
