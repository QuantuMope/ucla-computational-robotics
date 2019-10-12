import numpy as np
from state_action import State, Action

ROTATION_SIZE = 12
MOV_ACTION_SIZE = 3
ROT_ACTION_SIZE = 3
DEF_START = (1, 1, 6)
WIDTH = 8
HEIGHT = 8


class GridWorld():

    def __init__(self, error_prob=0, start_pos=DEF_START):
        self.width = WIDTH
        self.height = HEIGHT

        self.curr_state = State(start_pos[0], start_pos[1], start_pos[2])
        self.error_prob = error_prob
        self.rewards = self.init_rewards()
        self.gamma = 0.9

    def get_reward(self, x, y):
        return self.rewards[x, y]

    def init_rewards(self):
        reward_space = np.zeros((self.width, self.height))
        reward_space[0, :] = -100
        reward_space[7, :] = -100
        reward_space[:, 0] = -100
        reward_space[:, 7] = -100
        reward_space[1:4, 3] = -10
        reward_space[1, 5] = 1
        return reward_space

    def get_new_state(self, state, action):
        state_prob_map = self.get_prob(state, action)
        prob_cutoffs = [0]
        random_prob = np.random.uniform()
        i = 0
        for state, prob in state_prob_map.items():
            prob_cutoffs.append(prob+prob_cutoffs[i])
            i += 1
            if random_prob < prob_cutoffs[i]:
                return state

    # Returns Psa(s') given Pe, s, a, s'
    # s' = (x, y, rot)
    # a = (move_xy, rotate)
    def get_prob(self, state, action):
        probabilities = {}
        if action.move == 0:
            new_state = state.copy()
            new_state.rotate(action.rotate)
            probabilities[new_state.tuple()] = 1
        else:
            possible_errors = self.get_error_states(state)
            for poss_err in possible_errors:
                curr_state = poss_err[0]
                err_prob = poss_err[1]
                new_state = self.get_new_state_helper(curr_state, action)
                if new_state.tuple() in probabilities:
                    probabilities[new_state.tuple()] += err_prob
                else:
                    probabilities[new_state.tuple()] = err_prob

        assert abs(sum(probabilities.values()) - 1) < 0.01
        return probabilities
        # try:
        #     return probabilities[s_prime]
        # except IndexError():
        #     return 0

    def get_new_state_helper(self, state, action):
        rot = state.rotation
        orig = state.copy()
        new_pos = [state.x, state.y]
        if 2 <= rot <= 4:
            new_pos[0] += action.move
        elif 5 <= rot <= 7:
            new_pos[1] += -action.move
        elif 8 <= rot <= 10:
            new_pos[0] += -action.move
        else:
            new_pos[1] += action.move
        if self.check_map_edge(new_pos):
            new_state = State(new_pos[0], new_pos[1], orig.rotation)
        else:
            new_state = orig
        new_state.rotate(action.rotate)
        return new_state

    def check_map_edge(self, new_pos):
        if new_pos[0] < 0 or new_pos[0] > self.width-1 \
           or new_pos[1] < 0 or new_pos[1] > self.height-1:
            return False
        return True

    def get_error_states(self, state):
        err1_state, err2_state = state.copy(), state.copy()
        err1_state.rotate(1)
        err2_state.rotate(-1)
        return [(state, 1-2*self.error_prob),
                (err1_state, self.error_prob),
                (err2_state, self.error_prob)]

    # POLICY ITERATION
    def policy_iterate(self):
        pass

    def get_all_data(self, state, action):
        probabilities = self.get_prob(state, action)
        prob = []
        next_state = []
        rewards = []
        for ns, pr in probabilities.items():
            rewards.append(self.get_reward(ns[0], ns[1]))
            next_state.append(ns)
            prob.append(pr)
        return prob, next_state, rewards

    # VALUE ITERATION
    def value_iterate(self):
        tol = 0.001
        vf = np.zeros((self.width, self.height, ROTATION_SIZE))
        new_vf = vf.copy()
        while True:
            for x in range(self.width):
                for y in range(self.height):
                    for r in range(ROTATION_SIZE):
                        all_values = np.zeros((3, 3))
                        for move in range(-1, 2):
                            for rotate in range(-1, 2):
                                action = Action(move, rotate)
                                state = State(x, y, r)
                                prob, next_state, rewards = self.get_all_data(state ,action)
                                for i in range(len(prob)):
                                    all_values[move, rotate] += prob[i]*(rewards[i] + self.gamma*vf[next_state[i]])
                        new_vf[x, y, r] = np.amax(all_values)
            if np.all(np.abs(vf - new_vf) < tol):
                output = (np.round(new_vf, 2))
                for i in range(12):
                    print("------------------------- Rotation = {} ---------------------------------".format(i))
                    print(output[:, :, i])
                break
            vf = new_vf.copy()









def main():
    test = GridWorld()
    test.value_iterate()

if __name__ == "__main__":
    main()


