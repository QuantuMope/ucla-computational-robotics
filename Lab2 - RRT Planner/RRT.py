import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ParkingLot import ParkingLot
import utils

# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100


class RRT_Node:
    def __init__(self, initial_state):
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.theta = initial_state[2]
        self.parent = None
        self.actions = []
        self.path = []

    def get_xy(self):
        return np.array([self.x, self.y])

    def get_state(self):
        return self.x, self.y, self.theta

    def set_parent_path(self, parent_node, actions, path):
        self.parent = parent_node
        self.actions = actions
        self.path = path


class RRT_Robot:
    def __init__(self, parking_plot):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_rad = WHEEL_RADIUS

        self.curr_state = (self.x, self.y, self.theta)
        self.trajectory_ax = parking_plot[0]
        self.tree_ax = parking_plot[1]

        self.node_list = []
        self.config_space = []


    def calculate_config_space(self, obstacles):
        config_frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length)
        ts = self.trajectory_ax.transData
        for theta in np.linspace(0, 360, 3600):
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
            corner_to_center = np.round(corners - np.array([self.x, self.y])) * -1

            a, b, c, d = 3, 0, 1, 2
            if 90 <= theta < 180:
                a, b, c, d = 2, 3, 0, 1
            elif 180 <= theta < 270:
                a, b, c, d = 1, 2, 3, 0
            elif 270 <= theta < 360:
                a, b, c, d = 0, 1, 2, 3

            obstacle_boundaries = []
            for obstacle in obstacles:
                boundary = []
                bottom, top, left, right = utils.obstacle_to_corner(obstacle)
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
    def find_nearest_node(self, sample_point):
        shortest_dist = np.linalg.norm(self.RRT.get_xy() - sample_point)
        min_index = 0
        nearest_node = self.RRT
        for i, node in enumerate(self.node_list):
            dist = np.linalg.norm(node.get_xy() - sample_point)
            if dist < shortest_dist:
                shortest_dist = dist
                min_index = i
                nearest_node = node
        return nearest_node, min_index

    def drive(self, x, y, theta, left, right, dt=0.1):
        left_vel = utils.rpm_to_vel(left)
        right_vel = utils.rpm_to_vel(right)
        central_vel = (left_vel + right_vel) / 2
        dtheta = -(self.wheel_rad/self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(theta*2*np.pi/360) * dt
        dy = central_vel * np.cos(theta*2*np.pi/360) * dt

        theta = utils.add_angles(theta, dtheta)
        x += dx
        y += dy
        frame = plt.Rectangle((x-45, y-75), self.width, self.length)
        ts = self.trajectory_ax.transData
        tr = Affine2D().rotate_deg_around(x, y, -theta)
        t = tr + ts
        frame.set_transform(t)

        return dx, dy, int(dtheta)

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
                eq1 = -(self.wheel_rad / self.width) * (right_vel - left_vel) - theta_diff
                eq2 = ((left_vel + right_vel) / 2) * np.sin(theta_start * 2 * np.pi / 360) - x_diff
                eq3 = ((left_vel + right_vel) / 2) * np.cos(theta_start * 2 * np.pi / 360) - y_diff
                return np.abs([eq1, eq2, eq3])

            bnds = ([-157, 157])
            res = least_squares(equations, [1, 1], bounds=bnds)
            left_vel_input = res.x[0]
            right_vel_input = res.x[1]
            dx, dy, dtheta = self.drive(x_curr, y_curr, theta_curr,
                                        utils.vel_to_rpm(left_vel_input),
                                        utils.vel_to_rpm(right_vel_input), dt)

            x_curr += dx
            y_curr += dy
            theta_curr = utils.add_angles(theta_curr, dtheta)
            
            if self.check_collision((x_curr, y_curr, theta_curr)):
                control_inputs = None
                trajectory = None
                break

            control_inputs.append([utils.vel_to_rpm(left_vel_input), utils.vel_to_rpm(right_vel_input)])
            trajectory.append([x_curr, y_curr, theta_curr])
            t += dt

        new_node = RRT_Node((x_curr, y_curr, theta_curr))

        return control_inputs, trajectory, new_node

    # Question 2(c)
    def check_collision(self, state):
        if len(self.config_space) == 0:
            raise ValueError("Robot configuration space has not yet been computed. \n \
                              Please use calculate_config_space.")
        collision = False
        x, y, theta = state
        point = Point(x, y)
        config_space_plane = self.config_space[int(np.round(theta, 1)/0.1)]
        for obstacle in config_space_plane:
            coordinates2d = [(x, y) for x, y, theta in obstacle]
            polygon = Polygon(coordinates2d)
            collision = polygon.contains(point)
            if collision: break
        return collision

    def RRT_plan(self, initial_state, goal_state):
        self.node_list.append(RRT_Node(initial_state))
        node_counter = 0
        collision_counter = 0
        bottom, top, left, right = utils.obstacle_to_corner(goal_state)
        goal_polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])
        while True:
            sample_point = utils.sample_random_point()
            closest_point, closest_index = self.find_nearest_node(np.array([sample_point[0], sample_point[1]]))
            curr_node = self.node_list[closest_index]
            control_inputs, trajectory, new_node = self.generate_trajectory(curr_node.get_state(), sample_point)
            if control_inputs is None:
                print("Collision Count: {}".format(collision_counter))
                collision_counter += 1
                continue
            new_node.set_parent_path(curr_node, control_inputs, trajectory)
            self.node_list.append(new_node)
            if goal_polygon.contains(Point(new_node.x, new_node.y)):
                break
            node_counter += 1
            print("Node Count: {}".format(node_counter))

        # Plot the tree.
        for node in self.node_list:
            prev_node = node.parent
            if prev_node is None: continue
            self.tree_ax.plot(node.x, node.y, 'b.')
            self.tree_ax.plot([prev_node.x, node.x], [prev_node.y, node.y], 'g')

    def plot_path(self):
        curr_node = self.node_list[-1]
        trajectory = []
        while curr_node is not None:
            trajectory = curr_node.path + trajectory
            curr_node = curr_node.parent
        for state in trajectory:
            x, y, theta = state
            frame = plt.Rectangle((x-45, y-75), self.width, self.length,
                                  facecolor='cyan', linewidth=1, edgecolor='magenta')
            ts = self.trajectory_ax.transData
            tr = Affine2D().rotate_deg_around(x, y, -theta)
            t = tr + ts
            frame.set_transform(t)
            self.trajectory_ax.add_patch(frame)
            self.trajectory_ax.plot(x, y, 'bo')

def main():
    env = ParkingLot()

    initial_state = (125, 125, 0)
    robot = RRT_Robot(env.get_plots())
    robot.calculate_config_space(env.obstacles)
    # robot.plot_config_space()
    robot.RRT_plan(initial_state, env.goals)
    robot.plot_path()
    plt.show()



if __name__ == "__main__":
    main()
