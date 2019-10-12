import numpy as np

DEFAULT_START = (0,0)
ROTATION_SIZE = 12
MOV_ACTION_SIZE = 3
ROT_ACTION_SIZE = 3


class State:
    def __init__(self, x, y, rotation):
        self.x = x
        self.y = y
        self.rotation = rotation

    def change_rotate(self, rotate_change):
        new_rotation = self.rotation + rotate_change
        if new_rotation == -1:
            self.rotation = 11
        elif new_rotation == 12:
            self.rotation = 0
        else:
            self.rotation = new_rotation

    def copy(self):
        return State(self.x, self.y, self.rotation)

    def same_state(self, other_state):
        if self.x == other_state.x and self.y == other_state.y \
            and self.rotation == other_state.rotation:
            return True
        return False

    def tuple(self):
        return self.x, self.y, self.rotation


class Action:
    def __init__(self, move, rotate):
        if move < -1 or move > 1 or rotate > 11 or rotate < 0:
            raise ValueError("Improper move or rotate value input")
        self.actions = np.zeros((MOV_ACTION_SIZE, ROT_ACTION_SIZE))
        self.move = move
        self.rotate = rotate


class GridWorld():

    def __init__(self, width, height, error_prob, start_pos=DEFAULT_START):
        self.width = width
        self.height = height
        self.state_space = np.zeros((width, height, ROTATION_SIZE))
        self.action_space = np.zeros((MOV_ACTION_SIZE, ROT_ACTION_SIZE))

        self.state = State(start_pos[0], start_pos[1], 0)
        self.error_prob = error_prob

    def get_new_state(self, state, action, s_prime):
        _, state_prob_map = self.get_prob(state, action, s_prime)
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
    def get_prob(self, state, action, s_prime):
        probabilities = {}
        if action.move == 0:
            new_state = state.copy().change_rotate(action.rotate)
            probabilities[new_state.tuple()] = 1
        else:
            possible_errors = self.get_error_states(state)
            for poss_err in possible_errors:
                curr_state = poss_err[0]
                err_prob = poss_err[1]
                new_state = self.get_new_state_helper(curr_state, action)
                probabilities[new_state.tuple()] += err_prob

        assert abs(sum(probabilities.values()) - 1) < 0.01
        try:
            return probabilities[s_prime], probabilities
        except IndexError():
            return 0, probabilities

    def get_new_state_helper(self, state, action):
        rot = state.rotation
        orig = state.copy()
        new_pos = (state.x, state.y)
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
        new_state.change_rotate(action.rotate)
        return new_state

    def check_map_edge(self, new_pos):
        if new_pos[0] < 0 or new_pos[0] > self.width or new_pos[1] < 0 or new_pos[1] > self.height:
            return False
        return True

    def get_error_states(self, state):
        err1_state = state.copy().change_rotation(1)
        err2_state = state.copy().change_rotation(-1)
        return [(state, 1-2*self.error_prob),
                (err1_state, self.error_prob),
                (err2_state, self.error_prob)]

