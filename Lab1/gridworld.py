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

    def same_state(self, other_state):
        if self.x == other_state.x and self.y == other_state.y \
            and self.rotation == other_state.rotation:
            return True
        return False

class Action:
    def __init__(self, move, rotate):
        self.actions = np.zeros((MOV_ACTION_SIZE, ROT_ACTION_SIZE))
        self.move = move
        self.rotate = rotate

class GridWorld():

    def __init__(self, width, height, error_prob, start_pos=DEFAULT_START):
        self.state_space = np.zeros((width, height, ROTATION_SIZE))
        self.action_space = np.zeros((MOV_ACTION_SIZE, ROT_ACTION_SIZE))

        self.state = State(DEFAULT_START[0], DEFAULT_START[1], 0)

    # Returns Psa(s') given Pe, s, a, s'
    # s' = (x, y, rot)
    # a = (move_xy, rotate)
    def get_prob(self, action, s_prime):
        # check to see if s_prime is a reachable state
        if np.abs(s_prime.x-self.state.x) > 1 or np.abs(s_prime.y-self.state.y) > 1\
                or np.abs(s_prime.rotation-self.state.rotation) > 1: # solve rotation 11 and 0 case
            return 0
        if action.move == 0 and self.state.same_state(s_prime):
            return 1
        elif action.move == 0:
            return 0
        if action.move == 1V
        ### REST OF POSSIBLE



    # returns new x, y, and rot
    # pos_change should be -1 or 1
    def move_grid(self, state, action):
        # movement depends on robot rotation
        new_state =
        if 2 <= state[2] <= 4:
            new_state += pos_change
        elif 5 <= state[2] <= 7:
            self.position[1] += -pos_change
        elif 8 <= state[2] <= 10:
            self.position[0] += -pos_change
        else:
            self.position[1] += pos_change

        # makes sure robot does not go off grid
        if self.position[0] < 0:
            self.position[0] = 0
        elif self.position

    def act(self, rot_change, pos_change):
        # First possible error rotation
        if pos_change != 0:
            self.error_rotate()

    def error_rotate(self):
        random_prob = np.random.uniform()
        if random_prob < self.error_prob:
            self.change_rotation(1)
        elif self.error_prob < random_prob < 2*self.error_prob:
            self.change_rotation(-1)

    # change should be -1 or 1
    def change_rotation(self, change):
        self.rotation += change
        if self.rotation == 12:
            self.rotation = 0
        elif self.rotation == -1:
            self.rotation = 11
