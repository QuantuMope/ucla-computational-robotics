import numpy as np
import matplotlib.pyplot as plt
from utils import si, ai, new_state, get_error_states, init_policy

ROTATION_SIZE = 12
ACTION_SIZE = 4
DEF_START = (1, 1, 6)
GRID_WIDTH = 8
GRID_HEIGHT = 8


class GridWorld:

    def __init__(self, gamma=0.9, start_pos=DEF_START):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT

        self.gamma = gamma

        # Problem 1(a)
        self.state_space = self.init_state_space()
        self.Ns = len(self.state_space)

        # Problem 1(b)
        self.action_space = self.init_action_space()
        self.Na = len(self.action_space)

        self.reward_space = self.init_rewards()


    def init_state_space(self):
        """

        :return: State space as numpy array
        """
        state_space = []
        for x in range(self.width):
            for y in range(self.height):
                for r in range(ROTATION_SIZE):
                    state_space.append((x, y, r))
        return state_space

    def init_action_space(self):
        """

        :return:
        """
        action_space = []
        for move in range(-1, 2):
            for rotate in range(-1, 2):
                if move == 0 and rotate != 0:
                    continue
                action_space.append((move, rotate))
        return action_space

    def init_rewards(self):
        rs = np.zeros((self.width, self.height))
        rs[0][:] = -100
        rs[7][:] = -100
        rs[:, 0] = -100
        rs[:, 7] = -100
        rs[3, 4:7] = -10
        rs[5][6] = 1
        return rs

    def display_rewards(self):
        rs = np.swapaxes(self.reward_space, 0, 1)
        rs = np.flip(rs, 0)
        print(rs)

    # Problem 2(a)
    def get_reward(self, state):
        return self.reward_space[state[0], state[1]]

    def get_all_prob(self, state, action, pe):
        """

        :param state:
        :param action:
        :return: dictionary containing ...
                keys: all possible next_states, (x', y', r')
                prob: transition prob of next_state, Psa(s')
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

        assert abs(sum(probabilities.values()) - 1) < 0.001
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
        probabilities = self.get_all_prob(state, action, pe)
        probs = []
        next_states = []
        rewards = []
        for ns, pr in probabilities.items():
            probs.append(pr)
            next_states.append(ns)
            rewards.append(self.get_reward(ns))
        return probs, next_states, rewards

    # 3. POLICY ITERATION

    # Problem 3(b)
    def generate_trajectory(self, initial_state, policy, pe):
        fig, ax = plt.subplots(figsize=(self.width, self.height))
        ax.grid(color='black')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Robot Trajectory")
        plt.xlim((0, self.width))
        plt.ylim((0, self.height))
        Y = self.height-1
        X = self.width-1
        
        bottom = plt.Rectangle((0, 0), self.width, 1, color='red')
        top = plt.Rectangle((0, Y), self.width, 1, color='red')
        left = plt.Rectangle((0, 0), 1, self.height, color='red')
        right = plt.Rectangle((X, 0), 1, self.height, color='red')
        ax.add_patch(bottom), ax.add_patch(top), ax.add_patch(left), ax.add_patch(right)
        for x in range(self.width):
            for y in range(self.height):
                if x != 0 and x != X and y != 0 and y != Y:
                    continue
                plt.plot(x+0.5, y+0.5, markersize=20, marker='x', color='black')

        barrier = plt.Rectangle((3, 4), 1, 3, color='yellow')
        for y in range(4, 7):
            plt.plot(3.5, y+0.5, markersize=20, marker='_', color='black')
        goal = plt.Rectangle((5, 6), 1, 1, color='green')
        plt.plot(5.5, 6.5, markersize=20, marker='*', color='black')
        ax.add_patch(barrier), ax.add_patch(goal)

        trajectory = []
        curr_state = initial_state
        while True:
            trajectory.append(curr_state)
            action = policy[si(curr_state)]
            if action == (0, 0):
                break
            curr_state = self.get_new_state(curr_state, action, pe)

        old_x, old_y = trajectory[0][0]+0.5, trajectory[0][1]+0.5
        for state in trajectory:
            x = state[0] + 0.5
            y = state[1] + 0.5
            dx = np.sin((2*np.pi)*(state[2]*30/360)) * 0.5
            dy = np.cos((2*np.pi)*(state[2]*30/360)) * 0.5
            plt.plot(x, y, marker='o', markersize=10, color='black')
            plt.arrow(x, y, old_x-x, old_y-y, linestyle=':', color='blue')
            plt.arrow(x, y, dx, dy, head_width=0.05)
            old_x = x
            old_y = y


    # 4. VALUE ITERATION
    def value_iterate(self, pe, tol=0.0001):
        vf = np.zeros(self.Ns)
        new_vf = np.zeros(self.Ns)
        while True:
            for state in self.state_space:
                q_values = np.zeros(self.Na)
                for e, action in enumerate(self.action_space):
                    probs, next_states, rewards = self.get_all_data(state, action, pe)
                    for i in range(len(probs)):
                        q_values[e] += probs[i] * (rewards[i] + self.gamma * vf[si(next_states[i])]) 
                new_vf[si(state)] = np.amax(q_values)


            if np.all(np.abs(vf - new_vf) < tol):
                output = (np.round(new_vf, 3))
                output = output.reshape(8, 8, 12)
                print(output.shape)
                for i in range(12):
                    print("------------------------- Rotation = {} ---------------------------------".format(i))
                    print(np.flip(np.swapaxes(output[:, :, i], 0, 1), 0))
                break


            vf = new_vf.copy()



def main():
    test = GridWorld()
    test.generate_trajectory((1, 6, 6), init_policy(test.state_space), 0)
    plt.show()

    # test.value_iterate(pe=0)
    print("bp")


if __name__ == "__main__":
    main()
