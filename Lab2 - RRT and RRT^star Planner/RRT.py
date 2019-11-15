import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from ParkingLot import ParkingLot
from RRT_Node import RRT_Node
import utils

# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100


class RRT_Robot:
    """
    Two wheeled differential drive robot for fully observable
    continuous state and action space.

    State: robot pose --> (x position, y position, theta)
    Action: left and right wheel (-60 to 60 RPM)

    :param env_plot: the plots of the environment for drawing
                     RRT tree as well as robot trajectory.
    """
    def __init__(self, env_plot):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_radius = WHEEL_RADIUS

        self.trajectory_ax = env_plot[0]
        self.tree_ax = env_plot[1]

        self.node_list = []
        self.config_space_vertices = []
        self.config_space_polygons = []

    def compute_config_space(self, obstacles):
        """
        Computes the configuration space C of the environment by
        using the Minkowski sum of the robot's and obstacles' shapes.
        Computed for every 0.1 degree turn of the robot. Configuration
        space for each 0.1 degree is stored as an array of C_obs polygons.

        WORKS FOR RECTANGULAR OBSTACLES ONLY

        Reference:
        https://www.cs.cmu.edu/~motionplanning/lecture/Chap3-Config-Space_howie.pdf

        :param obstacles: array of environment obstacles (including walls)
                          containing obstacles in following format:
                          [lower left corner x coordinate,
                           lower left corner y coordinate,
                           width,
                           height]
        """
        center_x, center_y = 500, 500 # arbitrary values
        config_frame = plt.Rectangle((center_x-45, center_y-75), self.width, self.length)
        ts = self.trajectory_ax.transData

        for theta in np.linspace(0, 360, 3610):

            # Compute x, y coordinate of robot center given theta rotation using transform.
            tr = Affine2D().rotate_deg_around(center_x, center_y, -theta)
            t = tr + ts
            config_frame.set_transform(t)
            x, y = config_frame.get_x(), config_frame.get_y()

            # Corner coordinates of robot.
            # Index order: bottom left, top left, top right, bottom right
            robot_corners = np.array([[x, y],
                                      [x, y+self.length],
                                      [x+self.width, y+self.length],
                                      [x+self.width, y]])
            robot_corners = tr.transform(robot_corners)

            # Compute the x and y distance from each corner to center of robot.
            robot_ctc = np.round(robot_corners - np.array([center_x, center_y])) * -1

            # Depending on the theta value, different corners of the robot and obstacle
            # are used in computing the Minkowski sum polygon.
            a, b, c, d = 3, 0, 1, 2
            if 90 <= theta < 180:
                a, b, c, d = 2, 3, 0, 1
            elif 180 <= theta < 270:
                a, b, c, d = 1, 2, 3, 0
            elif 270 <= theta < 360:
                a, b, c, d = 0, 1, 2, 3

            # Compute the Minkowski sum polygon points for each obstacle.
            obstacle_vertices, obstacle_polygons = [], []
            for obstacle in obstacles:
                vertices = []
                bottom, top, left, right = utils.obstacle_to_corner(obstacle)
                vertices.append((left + robot_ctc[a][0], top + robot_ctc[a][1], theta))
                vertices.append((right + robot_ctc[a][0], top + robot_ctc[a][1], theta))
                vertices.append((right + robot_ctc[b][0], top + robot_ctc[b][1], theta))
                vertices.append((right + robot_ctc[b][0], bottom + robot_ctc[b][1], theta))
                vertices.append((right + robot_ctc[c][0], bottom + robot_ctc[c][1], theta))
                vertices.append((left + robot_ctc[c][0], bottom + robot_ctc[c][1], theta))
                vertices.append((left + robot_ctc[d][0], bottom + robot_ctc[d][1], theta))
                vertices.append((left + robot_ctc[d][0], top + robot_ctc[d][1], theta))

                obstacle_vertices.append(vertices)

                coordinates2d = [(x, y) for x, y, theta in vertices]
                obstacle_polygons.append(Polygon(coordinates2d))

            self.config_space_vertices.append(obstacle_vertices)
            self.config_space_polygons.append(obstacle_polygons)

    def plot_config_space(self):
        """
        Plot the configuration space in 3D space.
        """
        if len(self.config_space_vertices) == 0:
            raise RuntimeError("Configuration space has not been initialized. \n"
                               "Please run compute_config_space with environment obstacles.")
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim3d((0, 2000))
        ax.set_ylim3d((0, 1400))
        ax.set_zlim3d((0, 360))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Theta")
        plt.title("Configuration Space")
        for theta, all_obstacles in enumerate(self.config_space_vertices):
            color = 'steelblue'
            # Every 10 degrees, draw a black layer for improved visualization of contour.
            if theta % 10 == 0: color = 'black'
            for obstacle_verts in all_obstacles:
                ax.add_collection3d(Poly3DCollection([obstacle_verts], edgecolors=color))

    # Question 2(a)
    def _find_nearest_node(self, sample_point):
        """
        Helper function to find nearest node in current RRT tree to sample_point.
        Use the L2 norm of x, y, and theta to determine nearest.

        :param sample_point: numpy array [x, y, theta] of sample
        :return nearest_node: RRT_Node which is "nearest" to sample
        """
        nearest_node = self.node_list[0] # root of RRT tree
        shortest_dist = np.linalg.norm(nearest_node.get_state() - sample_point)
        for node in self.node_list:
            dist = np.linalg.norm(node.get_state() - sample_point)
            if dist < shortest_dist:
                shortest_dist = dist
                nearest_node = node
        return nearest_node

    def _drive(self, theta, left, right, dt=0.1):
        """
        Helper function to simulate robot driving given
        current theta and wheel RPMs. Uses differential drive kinematics.

        Reference:
        http://planning.cs.uiuc.edu/node659.html

        :param theta: robot theta
        :param left: left wheel angular velocity (rad/sec)
        :param right: right wheel angular velocity (rad/sec)
        :param dt: time step in seconds, default 0.1sec
        :return dx: travelled distance in x
        :return dy: travelled distance in y
        :return dtheta: rotation change (radians)
        """
        central_vel = self.wheel_radius * (left + right) / 2
        dtheta = -(self.wheel_radius / self.width) * (right - left) * dt
        dx = central_vel * np.sin(math.radians(theta)) * dt
        dy = central_vel * np.cos(math.radians(theta)) * dt

        return dx, dy, dtheta

    # Question 2(b)
    def _generate_trajectory(self, initial_state, goal_state, dt=0.1):
        """
        Generates a trajectory from initial_state attempting to reach target_state.
        Actions chosen by minimizing the least squares difference of initial and target
        state x, y, and theta. Carries out optimal action for certain time step and then
        recomputes action. Repeats this process for a total of one second.

        :param initial_state: numpy array [x, y, theta] of initial state
        :param goal_state: numpy array [x, y, theta] of goal state
        :param dt: time step, default 0.1 second

        If collision occurs during trajectory, None is returned for all three values

        :return actions: sequence of actions from initial_state to new_node
        :return trajectory: sequence of states from initial_state to new_node
        :return new_node: RRT_Node of final state
        """
        x_curr, y_curr, theta_curr = initial_state
        x_goal, y_goal, theta_goal = goal_state

        t = 0
        actions = []
        trajectory = [[x_curr, y_curr, theta_curr]]
        wheel_velocity_bounds = ([-6.3, 6.3]) # in rad/s

        """
        ------------------ Drive Policy ----------------------
        Solve system of equations. Minimize least squares.
        x = left wheel angular velocity (rad/s)
        y = right wheel angular velocity (rad/s)

        eqn1:  -(25/90) * (y - x) = theta_diff
        eqn2:   25 * (x + y)/2 * sin(theta) = x_diff
        eqn3:   25 * (x + y)/2 * cos(theta) = y_diff
        ------------------------------------------------------
        """

        def equations(p):
            left_vel, right_vel = p
            eq1 = -(self.wheel_radius / self.width) * (right_vel - left_vel) - math.radians(theta_diff)
            eq2 = self.wheel_radius * (left_vel + right_vel) / 2 * np.sin(math.radians(theta_curr)) - x_diff
            eq3 = self.wheel_radius * (left_vel + right_vel) / 2 * np.cos(math.radians(theta_curr)) - y_diff
            return np.abs([eq1, eq2, eq3])

        while t < 1:
            x_diff = x_goal - x_curr
            y_diff = y_goal - y_curr
            theta_diff = theta_goal - theta_curr

            res = least_squares(equations, [1, 1], bounds=wheel_velocity_bounds)
            left_vel_input, right_vel_input = res.x

            dx, dy, dtheta = self._drive(theta_curr, left_vel_input, right_vel_input, dt)

            x_curr += dx
            y_curr += dy
            theta_curr = utils.add_angles(theta_curr, math.degrees(dtheta))
            
            if self._check_collision((x_curr, y_curr, theta_curr)):
                actions = trajectory = None
                break

            actions.append([utils.vel_to_rpm(left_vel_input), utils.vel_to_rpm(right_vel_input)])
            trajectory.append([x_curr, y_curr, theta_curr])
            t += dt

        new_node = RRT_Node((x_curr, y_curr, theta_curr))

        return actions, trajectory, new_node

    # Question 2(c)
    def _check_collision(self, state):
        """
        Helper function that checks if the provided state occurs in collision space C_obs.

        :param state: numpy array [x, y, theta] of state
        :return: True if state is in C_obs, False otherwise
        """
        if len(self.config_space_polygons) == 0:
            raise RuntimeError("Robot configuration space has not yet been initialized. \n \
                                Please use calculate_config_space with environment obstacles.")
        collision = False
        x, y, theta = state
        state_point = Point(x, y)
        # Round current theta to nearest 0.1 degree and then divide by 0.1 to find proper index
        # in configuration space array.
        config_space_plane = self.config_space_polygons[int(np.round(theta, 1)/0.1)]
        for obstacle in config_space_plane:
            collision = obstacle.contains(state_point)
            if collision: break
        return collision

    # Question 2(d)
    def RRT_plan(self, initial_state, goal_state, max_iters=3000):
        """
        Robot planner that attempts to find a viable path free of obstacles from the provided
        initial state to the goal state using RRT algorithm.

        Planner ends when number of max iterations occurs or a path is found.

        :param initial_state: numpy array [x, y, theta] of initial state
        :param goal_state: numpy array [x, y, theta] of goal state
        :param max_iters: number of iterations before ending RRT algorithm
        """
        # Initialize root of RRT tree and goal area.
        self.node_list.append(RRT_Node(initial_state))
        node_counter = collision_counter = 0
        bottom, top, left, right = utils.obstacle_to_corner(goal_state)
        goal_polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])

        while True:
            # Sample a point in the state space. Every 50 nodes, sample from goal region.
            sample_point = utils.sample_random_point(0, 2000, 0, 1400, 0, 360)
            if node_counter % 25 == 0:
                sample_point = utils.sample_random_point(left, right, bottom, top+200, 170, 190)
                node_counter += 1 # Necessary to avoid infinite loop if a collision occurs
            # Resample if sample is in C_obs.
            if self._check_collision(sample_point): continue

            # Find nearest node and generate a trajectory driving towards sampled point.
            nearest_node = self._find_nearest_node(sample_point)
            actions, trajectory, new_node = self._generate_trajectory(nearest_node.get_state(), sample_point)

            # Check to see if a collision occurred during trajectory generation.
            if trajectory is None:
                print("Collision Count: {}".format(collision_counter))
                collision_counter += 1
                continue

            # If no collision, add new node to the RRT tree and store parent node and sequence of actions/states.
            new_node.set_parent(nearest_node, actions, trajectory)
            self.node_list.append(new_node)

            # End conditions.
            if goal_polygon.contains(Point(new_node.x, new_node.y)):
                print("Path Found!")
                break
            if node_counter > max_iters:
                print("Path could not be found after {} samples.".format(max_iters))
                break

            node_counter += 1
            print("Node Count: {}".format(node_counter))

    def plot_rrt_tree(self):
        """
        Plot the RRT tree.
        """
        for node in self.node_list:
            prev_node = node.parent
            if prev_node is None: continue
            self.tree_ax.plot([prev_node.x, node.x], [prev_node.y, node.y], 'magenta')
            self.tree_ax.plot(node.x, node.y, 'b.')

    def plot_path(self, animation=False):
        """
        Plot the trajectory of the robot.
        If a path was found, the trajectory shown will be the path to goal state.
        If a path was not found, a trajectory will be shown where the final node
        is the last node that was added to the RRT tree.

        :param animation: if set to True, plots the trajectory as an animation
        """
        curr_node = self.node_list[-1]
        trajectory = []

        # Reconstruct proper sequential ordering of the trajectory.
        while curr_node is not None:
            trajectory = curr_node.path + trajectory
            curr_node = curr_node.parent

        # Plot the robot frame for every dt, state by state.
        ts = self.trajectory_ax.transData
        for state in trajectory:
            x, y, theta = state
            frame = plt.Rectangle((x-45, y-75), self.width, self.length,
                                  facecolor='cyan', linewidth=1, edgecolor='magenta')
            tr = Affine2D().rotate_deg_around(x, y, -theta)
            t = tr + ts
            frame.set_transform(t)
            self.trajectory_ax.add_patch(frame)
            self.trajectory_ax.plot(x, y, 'bo', markersize=2)
            if animation:
                plt.pause(0.01)

    def _find_better_node(self, x_new, orig_parent, threshold=75):
        """
        Helper function for RRT* planner.
        After a trajectory is found, attempts to find a more suitable parent node defined
        by a lower minimum L2 norm (same as _find_nearest_node) as well as lower cost (less
        number of total actions taken).

        :param x_new: the new node that was found when driving from original nearest node
        :param orig_parent: the original nearest node that was used to drive to x_new
        :param threshold: threshold value for filtering out nodes that are too far away
        :return better_node: RRT_Node that is better than the original parent
        """
        better_node = orig_parent
        for node in self.node_list:
            if np.linalg.norm(node.get_state() - x_new.get_state()) < threshold and node.cost < better_node.cost:
                better_node = node
        return better_node

    def _rewire(self, threshold=350):
        """
        Rewire function for RRT* algorithm. Attempts to restructure the tree to obtain
        more optimal trajectories.

        :param threshold: threshold value for filtering out nodes that are too far away
                          from current node to attempt rewiring
        """
        all_states = []
        for node in self.node_list:
            all_states.append(node.get_state())
        all_states = np.array(all_states)

        rewire_counter = rewire_success = 0
        for node in self.node_list:
            rewire_counter += 1
            if node.parent is None: continue
            orig_cost = node.cost
            orig_parent = better_node = node.parent

            if rewire_counter % 50 == 0: print("Rewire Check for Node: {}".format(rewire_counter))

            # Find all nodes that are within the threshold and then find the best node out of them.
            candidates = np.argwhere(np.sqrt(np.sum((all_states - node.get_state())**2, axis=1)) < threshold)
            for i in candidates:
                if self.node_list[i[0]].cost < better_node.cost:
                    better_node = self.node_list[i[0]]

            # If the best node is the original parent, continue.
            if better_node.equals(orig_parent): continue

            rewire_actions, rewire_trajectory = [], []
            rewire_cost = better_node.cost
            new_node = better_node

            # Attempt to drive from the better node to the original node in order to find a valid
            # trajectory and establish a connection. Repeat for multiple 1 second trajectories until
            # attempted rewiring becomes suboptimal.
            while True:
                actions, trajectory, new_node = self._generate_trajectory(new_node.get_state(), node.get_state())

                # If cost of node increases from this path or a collision occurs, cancel rewiring attempt.
                if rewire_cost > orig_cost or trajectory is None: break

                rewire_cost += len(actions)
                rewire_actions = rewire_actions + actions
                rewire_trajectory = rewire_trajectory + trajectory

                # From the proposed better node, if a path is found to a state that is within 3mm x, 3mm y, and 3
                # degrees theta from the original node, then set the parent of the original node to the better node.
                # This tolerance is necessary since we are in a continuous state system and for a rewire to work,
                # we must reconnect to already existing nodes.
                if np.all(np.abs(np.array([new_node.x-node.x, new_node.y-node.y, new_node.theta-node.theta])) < 3):
                    node.set_parent(better_node, rewire_actions, rewire_trajectory)
                    rewire_success += 1
                    print("Rewired Count: {}".format(rewire_success))
                    break

    def RRT_star_plan(self, initial_state, goal_state, max_iters=2000):
        """
        RRT* Planner
        Optimized version of RRT planning algorithm with some key differences:

        1. Once a new node is found, instead of connecting to the parent node right away, the RRT tree is
           searched for a more optimal parent node.
        2. After a path is found, the RRT tree is attempted to be rewired.

        More extensive comments can be found in the RRT_plan function.

        Computationally more expensive than RRT algorithm, therefore max iterations
        is set lower. Trade off between finding an optimal path and planner having
        to give up sooner than RRT planner due to computational complexity.
        """
        self.node_list.append(RRT_Node(initial_state))
        node_counter = collision_counter = 0
        bottom, top, left, right = utils.obstacle_to_corner(goal_state)
        goal_polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom)])

        while True:
            sample_point = utils.sample_random_point(0, 2000, 0, 1400, 0, 360)
            if node_counter % 25 == 0:
                sample_point = utils.sample_random_point(left, right, bottom, top+200, 170, 190)
                node_counter += 1

            if self._check_collision(sample_point): continue

            # Find new node.
            orig_nearest_node = self._find_nearest_node(sample_point)
            actions, trajectory, new_node = self._generate_trajectory(orig_nearest_node.get_state(), sample_point)
            if trajectory is None:
                collision_counter += 1
                print("Collision Count: {}".format(collision_counter))
                continue

            # Check for better possible node for parent to new node.
            better_node = self._find_better_node(new_node, orig_nearest_node)

            # If a better node is found, attempt to find a path from better node to new node instead.
            if not better_node.equals(orig_nearest_node):
                actions, trajectory, new_node = self._generate_trajectory(better_node.get_state(), new_node.get_state())
                if trajectory is None:
                    collision_counter += 1
                    print("Collision Count: {}".format(collision_counter))
                    continue

            node_counter += 1
            print("Node Count: {}".format(node_counter))

            new_node.set_parent(better_node, actions, trajectory)
            self.node_list.append(new_node)

            if goal_polygon.contains(Point(new_node.x, new_node.y)):
                print("Path Found!")
                break
            if node_counter > max_iters:
                print("Path could not be found after {} samples.".format(max_iters))
                break

        # Once RRT* algorithm finishes, rewire the tree.
        # Tree rewiring is done only at the very end due to computational expense.
        self._rewire()


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
