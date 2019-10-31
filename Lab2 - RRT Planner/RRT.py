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
        self.cost = 0
        self.actions = []
        self.path = []

    def get_state(self):
        return np.array([self.x, self.y, self.theta])

    # def get_state(self):
    #     return self.x, self.y, self.theta

    def set_parent(self, parent_node, actions, path):
        self.parent = parent_node
        self.actions = actions
        self.path = path
        self.cost = self.parent.cost + len(self.actions)

    def equals(self, other_node):
        return self.get_state() == other_node.get_state()

class RRT_Robot:
    def __init__(self, parking_plot):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_rad = WHEEL_RADIUS

        self.trajectory_ax = parking_plot[0]
        self.tree_ax = parking_plot[1]

        self.node_list = []
        self.config_space = []

    def compute_config_space(self, obstacles):
        center_x, center_y = 500, 500 # arbitrary values
        config_frame = plt.Rectangle((center_x-45, center_y-75), self.width, self.length)
        ts = self.trajectory_ax.transData
        for theta in np.linspace(0, 360, 3610):
            tr = Affine2D().rotate_deg_around(center_x, center_y, -theta)
            t = tr + ts
            config_frame.set_transform(t)
            x, y = config_frame.get_x(), config_frame.get_y()

            # bottom left, top left, top right, bottom right
            corners = np.array([[x, y],
                                [x, y+self.length],
                                [x+self.width, y+self.length],
                                [x+self.width, y]])
            corners = tr.transform(corners)
            corner_to_center = np.round(corners - np.array([center_x, center_y])) * -1

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
            if theta % 10 == 0: color = 'black'
            for obstacle_verts in all_obstacles:
                ax.add_collection3d(Poly3DCollection([obstacle_verts], edgecolors=color))

    # Question 2(a)
    def _find_nearest_node(self, sample_point):
        root = self.node_list[0]
        shortest_dist = np.linalg.norm(root.get_state() - sample_point)
        min_index = 0
        nearest_node = root
        for i, node in enumerate(self.node_list):
            dist = np.linalg.norm(node.get_state() - sample_point)
            if dist < shortest_dist:
                shortest_dist = dist
                min_index = i
                nearest_node = node
        return nearest_node, min_index

    def _drive(self, theta, left, right, dt=0.1):
        left_vel = utils.rpm_to_vel(left)
        right_vel = utils.rpm_to_vel(right)
        central_vel = (left_vel + right_vel) / 2
        dtheta = -(self.wheel_rad/self.width) * (right_vel - left_vel) * dt
        dx = central_vel * np.sin(theta*2*np.pi/360) * dt
        dy = central_vel * np.cos(theta*2*np.pi/360) * dt

        return dx, dy, dtheta

    # Question 2(b)
    def _generate_trajectory(self, initial_state, target_state, dt=0.1):
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
            Solve system of equations. Minimize least squares.
            x = left wheel velocity
            y = right wheel velocity
            
            eqn1:  -(25/90) * (x - y) = theta_diff
            eqn2:   (x + y)/2 * sin(theta) = x_diff
            eqn3:   (x + y)/2 * cos(theta) = x_diff
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
            dx, dy, dtheta = self._drive(theta_curr,
                                         utils.vel_to_rpm(left_vel_input),
                                         utils.vel_to_rpm(right_vel_input), dt)

            x_curr += dx
            y_curr += dy
            theta_curr = utils.add_angles(theta_curr, dtheta)
            
            if self._check_collision((x_curr, y_curr, theta_curr)):
                control_inputs = None
                trajectory = None
                break

            control_inputs.append([utils.vel_to_rpm(left_vel_input), utils.vel_to_rpm(right_vel_input)])
            trajectory.append([x_curr, y_curr, theta_curr])
            t += dt

        new_node = RRT_Node((x_curr, y_curr, theta_curr))

        return control_inputs, trajectory, new_node

    # Question 2(c)
    def _check_collision(self, state):
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

    # Question 2(d)
    def RRT_plan(self, initial_state, goal_state, max_iters=3000):
        self.node_list.append(RRT_Node(initial_state))
        node_counter = 0
        collision_counter = 0
        bottom, top, left, right = utils.obstacle_to_corner(goal_state)
        goal_polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])
        while True:
            sample_point = utils.sample_random_point(0, 2000, 0, 1400, 0, 360)
            if node_counter % 50 == 0:
                sample_point = utils.sample_random_point(left, right, bottom, top+200, 170, 190)
                node_counter += 1
            if self._check_collision(sample_point): continue
            closest_node, closest_index = self._find_nearest_node(np.array([sample_point[0], sample_point[1], sample_point[2]]))
            control_inputs, trajectory, new_node = self._generate_trajectory(closest_node.get_state(), sample_point)
            if control_inputs is None:
                print("Collision Count: {}".format(collision_counter))
                collision_counter += 1
                continue
            new_node.set_parent(closest_node, control_inputs, trajectory)
            self.node_list.append(new_node)
            if goal_polygon.contains(Point(new_node.x, new_node.y)):
                print("Path Found!")
                break
            if node_counter > max_iters:
                print("Path could not be found after {} samples.".format(max_iters))
                break
            node_counter += 1
            print("Node Count: {}".format(node_counter))

    def plot_rrt_tree(self):
        for node in self.node_list:
            prev_node = node.parent
            if prev_node is None: continue
            self.tree_ax.plot([prev_node.x, node.x], [prev_node.y, node.y], 'magenta')
            self.tree_ax.plot(node.x, node.y, 'b.')

    def plot_path(self, animation=False):
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
            self.trajectory_ax.plot(x, y, 'bo', markersize=2)
            if animation:
                plt.pause(0.01)

    def _find_better_node(self, x_new, orig_parent, threshold=75):
        better_node = orig_parent
        for node in self.node_list:
            if np.abs(np.linalg.norm(node.get_state() - x_new.get_state())) < threshold and node.cost < better_node.cost:
                better_node = node
        return better_node

    def rewire(self, threshold=350):
        all_states = []
        for node in self.node_list:
            all_states.append(node.get_state())
        all_states = np.array(all_states)

        rewire_check = 0
        rewire_counter = 0
        for node in self.node_list:
            rewire_check += 1
            if node.parent is None: continue
            orig_parent = node.parent
            orig_cost = node.cost
            better_node = node.parent
            if rewire_check % 50 == 0: print("Rewire Check for Node: {}".format(rewire_check))
            print("gets_caught_earlier")
            candidates = np.argwhere(np.sqrt(np.sum((all_states - node.get_state())**2, axis=1)) < threshold)
            for i in candidates:
                print("gets_caught")
                if self.node_list[i[0]].cost < better_node.cost:
                    better_node = self.node_list[i[0]]
            # if better_node.equals(orig_parent): continue
            rewire_control = []
            rewire_trajectory = []
            rewire_cost = better_node.cost
            new_node = better_node
            while True:
                print("enters loop")
                control_inputs, trajectory, new_node = self._generate_trajectory(new_node.get_state(), node.get_state())
                if rewire_cost > orig_cost or control_inputs is None: break
                rewire_cost += len(control_inputs)
                rewire_control = rewire_control + control_inputs
                rewire_trajectory = rewire_trajectory + trajectory
                print("This is broken")
                if np.all(np.abs(np.array([new_node.x-node.x, new_node.y-node.y, new_node.theta-node.theta])) < 3):
                    node.set_parent(better_node, rewire_control, rewire_trajectory)
                    node.cost = orig_cost
                    rewire_counter += 1
                    print("Rewired Count: {}".format(rewire_counter))
                    break

    def RRT_star_plan(self, initial_state, goal_state, max_iters=10):
        """
        Optimized version of RRT planning algorithm with key differences.
        1. New nodes are connected to closest node after trajectory not before.
        2. After a new node is created, all nodes are reconnected for shortest path.
        """
        self.node_list.append(RRT_Node(initial_state))
        node_counter = 0
        collision_counter = 0
        better_node_counter = 0
        bottom, top, left, right = utils.obstacle_to_corner(goal_state)
        goal_polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])
        while True:
            sample_point = utils.sample_random_point(0, 2000, 0, 1400, 0, 360)
            if node_counter % 50 == 0:
                sample_point = utils.sample_random_point(left, right, bottom, top+200, 170, 190)
                node_counter += 1
            if self._check_collision(sample_point): continue
            closest_node, closest_index = self._find_nearest_node(np.array([sample_point[0], sample_point[1], sample_point[2]]))
            control_inputs, trajectory, new_node = self._generate_trajectory(closest_node.get_state(), sample_point)
            if control_inputs is None:
                collision_counter += 1
                # print("Collision Count: {}".format(collision_counter))
                continue

            # Check for better possible node.
            better_node = self._find_better_node(new_node, closest_node)
            control_inputs, trajectory, new_node = self._generate_trajectory(better_node.get_state(), new_node.get_state())
            if control_inputs is None:
                collision_counter += 1
                print("Collision Count: {}".format(collision_counter))
                continue

            better_node = new_node
            node_counter += 1
            print("Node Count: {}".format(node_counter))

            new_node.set_parent(better_node, control_inputs, trajectory)
            self.node_list.append(new_node)
            if goal_polygon.contains(Point(new_node.x, new_node.y)):
                print("Path Found!")
                break
            if node_counter + better_node_counter > max_iters:
                print("Path could not be found after {} samples.".format(max_iters))
                break

        self.rewire()



def main():
    env = ParkingLot()

    initial_state = (125, 125, 0)
    robot = RRT_Robot(env.get_plots())
    robot.compute_config_space(env.obstacles)
    # robot.plot_config_space()
    robot.RRT_plan(initial_state, env.goals)
    # robot.RRT_star_plan(initial_state, env.goals)
    robot.plot_rrt_tree()
    robot.plot_path()
    plt.show()


if __name__ == "__main__":
    main()
