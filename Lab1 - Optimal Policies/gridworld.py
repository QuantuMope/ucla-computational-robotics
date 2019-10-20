import numpy as np
import matplotlib.pyplot as plt
import time
from utils import si, new_state, get_error_states, init_policy

ROTATION_SIZE = 12
ACTION_SIZE = 7
GRID_WIDTH = 8
GRID_HEIGHT = 8


class GridWorld:
    """
        Class capable of simulating an MDP system.
        Creates an 8x8 grid world.
    """

    def __init__(self, scenario):
        """

        :param scenario:
        """
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT

        self.state_space, self.Ns = self.init_state_space()

        self.action_space, self.Na = self.init_action_space()

        self.reward_space = self.init_rewards(scenario)

    # Problem 1(a)
    def init_state_space(self):
        """
        Initializes the state space according to grid width
        grid height, and possible rotational orientations.

        States are represented as tuples (x, y, r) where
        x = x-position ranging from (0-7)
        y = y-position ranging from (0-7)
        r = rotational orientation ranging from (0-11)

        :return state_space: array containing each possible
                             state for the mdp system
        :return Ns: size of the state space
        """
        state_space = []
        for x in range(self.width):
            for y in range(self.height):
                for r in range(ROTATION_SIZE):
                    state_space.append((x, y, r))
        Ns = len(state_space)
        return state_space, Ns

    # Problem 1(b)
    def init_action_space(self):
        """
        Initializes the action space.

        Actions are represented as tuples (move, rotate) where
        move = translational movement, can take on values
               -1 ==> move backwards
                0 ==> stay still
                1 ==> move forward
        rotate = rotational change, can take on values
               -1 ==> rotate counter-clockwise
                0 ==> don't rotate
                1 ==> rotate clockwise

        No rotation can be taken when robot decides to stay still.

        :return action_space: array containing each possible
                              action for the mdp system
        :return Na: size of action space
        """
        action_space = []
        for move in range(-1, 2):
            for rotate in range(-1, 2):
                if move == 0 and rotate != 0:
                    continue
                action_space.append((move, rotate))
        Na = len(action_space)
        return action_space, Na

    def init_rewards(self, scenario):
        """
        Initializes the rewards for the environment.

        :param scenario: string input to identify which
                         state(s) should become the goal
                         state(s)
        :return rs: numpy matrix storing the proper reward
                    for each state in the environment's state
                    space
        """
        if scenario is not "12" and scenario is not "34":
            raise ValueError("Invalid scenario input. Scenario should be string '12' or '34'.")
        rs = np.zeros((self.width, self.height, ROTATION_SIZE))
        rs[0][:][:] = -100
        rs[7][:][:] = -100
        rs[:, 0][:] = -100
        rs[:, 7][:] = -100
        rs[3, 4:7][:] = -10
        if scenario == "12":
            rs[5][6][:] = 1
        elif scenario == "34":
            rs[5][6][6] = 1
        return rs


    # Problem 2(a)
    def get_reward(self, state):
        """
        Simply returns the reward value at a certain state.

        :param state:
        :return: reward value
        """
        return self.reward_space[state[0], state[1], state[2]]

    def get_all_prob(self, state, action, pe):
        """

        :param state:
        :param action:
        :param pe: error probability ranging from
        :return: dictionary containing {s : p}
                 keys: all possible next_states, (x', y', r')
                 values: transition prob of next_state, Psa(s')
        """
        probabilities = {}
        if action[0] == 0:
            probabilities[state] = 1
        else:
            possible_errors = get_error_states(state, pe)
            for poss_err in possible_errors:
                curr_state = poss_err[0]
                err_prob = poss_err[1]
                new_s = new_state(curr_state, action)
                if new_s in probabilities:
                    probabilities[new_s] += err_prob
                else:
                    probabilities[new_s] = err_prob

        # Check to make sure probabilities sum up to 1.
        assert abs(sum(probabilities.values()) - 1) < 0.0001
        return probabilities

    # Problem 1(c)
    def get_prob(self, state, action, next_state, pe):
        """


        :param state:
        :param action:
        :param next_state:
        :param pe:
        :return:
        """
        probabilities = self.get_all_prob(state, action, pe)
        try:
            return probabilities[next_state]
        except IndexError():
            return 0

    # Problem 1(d)
    def get_new_state(self, state, action, pe):
        """

        :param state:
        :param action:
        :param pe:
        :return next_state:
        """
        probabilities = self.get_all_prob(state, action, pe)
        prob_cutoffs = [0]
        random_prob = np.random.uniform()
        i = 0
        for state, prob in probabilities.items():
            prob_cutoffs.append(prob+prob_cutoffs[i])
            i += 1
            if random_prob < prob_cutoffs[i]:
                return state

    def get_all_data(self, state, action, pe):
        """
        Helper function that neatly returns three lists
        containing the possible next states and their
        respective probabilities and rewards.

        :param state: starting state tuple (x, y, r)
        :param action: action tuple (move, rotate)
        :param pe: error probability

        Each next state's probability and reward are
        all linked by the same index in their own
        respective lists.

        :return probs: list of probabilities for next states
        :return next_states: list of possible next states
        :return rewards: list of rewards for next states
        """
        all_possible_states = self.get_all_prob(state, action, pe)
        probs = []
        next_states = []
        rewards = []
        for ns, pr in all_possible_states.items():
            probs.append(pr)
            next_states.append(ns)
            rewards.append(self.get_reward(ns))
        return probs, next_states, rewards

    # 3. POLICY ITERATION

    # Problem 3(b) and 3(c)
    def generate_trajectory(self, initial_state, policy, pe, vi_or_pi):
        """
        Function generates a trajectory following a provided policy
        starting at a provided initial state. Trajectory becomes
        stochastic given an error probability > 0.

        Trajectory is then plotted onto a grid representing the 8x8 gridworld.
        A black circle represents the robot with an arrow pointing out of its
        center representing its current heading value. A green dash line
        outlines the trajectory with numbers declaring the discrete time step
        that the position was encountered in.

        :param initial_state: starting state tuple (x, y, r)
        :param policy: numpy array mapping states to actions
        :param pe: error probability
        :param vi_or_pi: String value indicating method used to generate
                         the provided policy. Used to determine the proper
                         heading for trajectory plot. Values should either be
                         "vi" for value iteration,
                         "pi" for policy iteration, or
                         "ip" for initial policy
        :return trajectory: array containing all experienced states
        """

        # Translate vi_or_pi input into the proper method.
        if vi_or_pi not in ["vi", "pi", "ip"]:
            raise ValueError("vi_or_pi should either be string vi, pi, or ip")
        if vi_or_pi is "ip":
            method = "Initial Policy"
        elif vi_or_pi is "vi":
            method = "Value Iteration"
        else:
            method = "Policy Iteration"

        # Declare base parameters of the gridworld plot.
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.grid(color='black')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Robot Trajectory with pe={}% using {}".format(pe, method))
        plt.xlim((0, self.width))
        plt.ylim((0, self.height))
        Y = self.height-1
        X = self.width-1

        # Generate red rectangles resembling grid locations with -100 reward.
        bottom = plt.Rectangle((0, 0), self.width, 1, color='red')
        top = plt.Rectangle((0, Y), self.width, 1, color='red')
        left = plt.Rectangle((0, 0), 1, self.height, color='red')
        right = plt.Rectangle((X, 0), 1, self.height, color='red')
        ax.add_patch(bottom), ax.add_patch(top), ax.add_patch(left), ax.add_patch(right)
        # Add an X marker at each -100 reward location.
        for x in range(self.width):
            for y in range(self.height):
                if x != 0 and x != X and y != 0 and y != Y:
                    continue
                plt.plot(x+0.5, y+0.5, markersize=20, marker='x', color='black')

        # Create a yellow rectangle resembling grid locations with -10 reward.
        barrier = plt.Rectangle((3, 4), 1, 3, color='yellow')
        # Add a - marker at each -10 reward location.
        for y in range(4, 7):
            plt.plot(3.5, y+0.5, markersize=20, marker='_', color='black')

        # Create a green square with a star marker for the goal location with +1 reward.
        goal = plt.Rectangle((5, 6), 1, 1, color='lime')
        plt.plot(5.5, 6.5, markersize=20, marker='*', color='black')
        ax.add_patch(barrier), ax.add_patch(goal)

        # Generate a trajectory array following the provided policy.
        trajectory = []
        curr_state = initial_state
        while True:
            trajectory.append(curr_state)
            action = policy[si(curr_state)]
            # End trajectory if goal state is reached.
            if action == (0, 0):
                break
            # Generates a next state stochastically with provided error probability.
            curr_state = self.get_new_state(curr_state, action, pe)

        # Plot the trajectory.
        old_x, old_y = trajectory[0][0]+0.5, trajectory[0][1]+0.5
        for i, state in enumerate(trajectory):
            x = state[0] + 0.5
            y = state[1] + 0.5
            dx = np.sin((2*np.pi)*(state[2]*30/360)) * 0.5
            dy = np.cos((2*np.pi)*(state[2]*30/360)) * 0.5
            # Represent robot as a black circle.
            plt.plot(x, y, marker='o', markersize=10, color='black')
            # Represent robot heading as an arrow.
            plt.arrow(x, y, dx, dy, head_width=0.05, color='black')
            # Plot trajectory as green dashed line and label each discrete time step.
            plt.arrow(x, y, old_x-x, old_y-y, linestyle=':', width=0.015, color='forestgreen')
            plt.text(x+dx, y+dy, str(i), fontsize=12)
            old_x = x
            old_y = y

        return trajectory

    # Problem 3(d)
    def evaluate_policy(self, policy, pe, gamma=0.9, tol=0.0001):
        """
        Part 1 of Policy Iteration: Policy Evaluation

        Generates the value function for a provided policy with
        the provided error probability and discount factor.

        :param policy: numpy array mapping states to actions
        :param pe: error probability
        :param gamma: discount factor
        :param tol: tolerance value used to dictate when policy evaluation ends
        :return new_vf: value function for the provided policy
        """
        vf = np.zeros(self.Ns)

        while True:
            new_vf = np.zeros(self.Ns)
            for state in self.state_space:
                probs, next_states, rewards = self.get_all_data(state, policy[si(state)], pe)
                for i in range(len(probs)):
                    # Assign the expected sum of discounted rewards to each state.
                    new_vf[si(state)] += probs[i] * (rewards[i] + gamma * vf[si(next_states[i])])

            # End policy evaluation if old value function differs from the new value function
            # by less than a provided tolerance for all states.
            if np.all(np.abs(vf - new_vf) < tol):
                break
            vf = new_vf.copy()

        return new_vf

    # Problem 3(f)
    def improve_policy(self, vf, pe, gamma=0.9):
        """
        Part 2 of Policy Iteration: Policy Improvement

        Using the value function of the previous policy,
        generate a new and improved policy.

        :param vf: value function of previous policy
        :param pe: error probability
        :param gamma: discount factor
        :return new_policy: an improved policy
        """
        new_policy = [None]*self.Ns
        for state in self.state_space:
            q_values = np.zeros(self.Na)
            for e, action in enumerate(self.action_space):
                probs, next_states, rewards = self.get_all_data(state, action, pe)
                for i in range(len(probs)):
                    # Generate the Q values for each action.
                    q_values[e] += probs[i] * (rewards[i] + gamma * vf[si(next_states[i])])
            # Assign the argmax of the Q values (i.e. best action) to the new policy.
            new_policy[si(state)] = self.action_space[np.argmax(q_values)]

        # Make sure all states are assigned an action.
        assert None not in new_policy
        return new_policy

    # Problem 3(g)
    def policy_iterate(self, pe, gamma=0.9, tol=0.0001, print_vf=False):
        """
        Uses functions policy_eval and improve_policy to
        employ policy iteration.

        :param pe: error probability
        :param gamma: discount factor
        :param tol:
        :return optimal_policy: numpy array that contains a optimal
                                action for each state
        :return optimal_vf: the optimal value function
        """
        init_pol = init_policy(self.state_space)
        vf = self.evaluate_policy(init_pol, pe, gamma, tol)
        old_vf = vf.copy()
        while True:
            new_policy = self.improve_policy(vf, pe, gamma)
            vf = self.evaluate_policy(new_policy, pe, gamma, tol)

            if np.all(np.abs(vf - old_vf) < tol):
                if print_vf:
                    output = (np.round(vf, 3))
                    output = output.reshape(8, 8, 12)
                    for i in range(12):
                        print("------------------------- Rotation = {} ---------------------------------".format(i))
                        print(np.flip(np.swapaxes(output[:, :, i], 0, 1), 0))
                break
            old_vf = vf.copy()
        optimal_policy, optimal_vf = new_policy, vf
        return optimal_policy, optimal_vf

    # 4. VALUE ITERATION
    # Problem 4(a)
    def value_iterate(self, pe, gamma=0.9, tol=0.0001, print_vf=False):
        vf = np.zeros(self.Ns)
        new_vf = np.zeros(self.Ns)
        while True:
            for state in self.state_space:
                q_values = np.zeros(self.Na)
                for e, action in enumerate(self.action_space):
                    probs, next_states, rewards = self.get_all_data(state, action, pe)
                    for i in range(len(probs)):
                        q_values[e] += probs[i] * (rewards[i] + gamma * vf[si(next_states[i])])
                new_vf[si(state)] = np.amax(q_values)

            if np.all(np.abs(vf - new_vf) < tol):
                if print_vf:
                    output = (np.round(new_vf, 3))
                    output = output.reshape(8, 8, 12)
                    for i in range(12):
                        print("------------------------- Rotation = {} ---------------------------------".format(i))
                        print(np.flip(np.swapaxes(output[:, :, i], 0, 1), 0))
                break
            vf = new_vf.copy()
        optimal_vf = vf
        optimal_policy = self.improve_policy(optimal_vf, pe, gamma)
        return optimal_policy, optimal_vf

def main():
    # =============================== PARAM INTERFACE ====================================
    """
        Simple interface to simulate trajectories.
        pe {0... 0.5}
        scenario {"12", "34}
    """
    pe = 0.25
    runs = 1
    generate_traj = True
    scenario = "34"
    # ====================================================================================

    # Note: Nothing below this should be touched!
    vi_times = []
    pi_times = []

    if pe > 0.5 or pe < 0:
        raise ValueError("Invalid error probability input. Pe should be 0 <= Pe <= 0.5")

    for i in range(runs):

        env = GridWorld(scenario)
        init_pol = init_policy(env.state_space)
        init_state = (1, 6, 6)
        if generate_traj:
            env.generate_trajectory(init_state, policy=init_pol, pe=pe, vi_or_pi="ip")
        # plt.show()

        # Problem 3(e)
        init_policy_vf = env.evaluate_policy(init_pol, pe=pe)

        # Problem 3(h) and 3(i)
        start = time.time()
        pi_pol, pi_vf = env.policy_iterate(pe=pe)
        end = time.time()
        print("Policy Iteration took {} seconds.".format(np.round(end-start, 3)))
        pi_times.append(end-start)
        if generate_traj:
            pi_traj = env.generate_trajectory(init_state, policy=pi_pol, pe=pe, vi_or_pi="pi")
        # plt.show()

        # Problem 4(b)
        start = time.time()
        vi_pol, vi_vf = env.value_iterate(pe=pe)
        end = time.time()
        print("Value Iteration took {} seconds.".format(np.round(end-start, 3)))
        vi_times.append(end-start)
        if generate_traj:
            vi_traj = env.generate_trajectory(init_state, policy=vi_pol, pe=pe, vi_or_pi="vi")

    avg_pi_time = np.array(pi_times).mean()
    avg_vi_time = np.array(vi_times).mean()
    print("Policy Iteration took an average of {} seconds measured over {} runs.".format(avg_pi_time, runs))
    print("Value Iteration took an average of {} seconds measured over {} runs.".format(avg_vi_time, runs))

    if generate_traj:
        plt.show()

if __name__ == "__main__":
    main()
