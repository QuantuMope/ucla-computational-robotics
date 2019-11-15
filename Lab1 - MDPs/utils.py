import numpy as np

"""
    Contains various helper functions for main
    functions in gridworld.py
"""


def si(state):
    """
    Helper function that converts state tuples to
    indexes for proper indexing of the state space.

    :param state: state tuple consisting of (x, y, h)
    :return: index of the state in the state space
    """
    return state[2] + 12 * state[1] + 96 * state[0]


def get_error_states(state, pe):
    """
    Helper function that outputs a list of possible
    states due to a given error probability.

    :param state: initial state tuple (x, y, h)
    :param pe: error probability
    :return: list of tuples for each possible error
             state [(state tuple, probability of occurrence)]
    """

    err1_state = new_state(state, (0, 1))
    err2_state = new_state(state, (0, -1))
    return [(state, 1-2*pe),
            (err1_state, pe),
            (err2_state, pe)]


def new_state(state, action):
    """
    Helper function that outputs the resultant
    state given an initial state and action taken.

    :param state: initial state tuple (x, y, h)
    :param action: action tuple (move, rotate)
    :return new_s: resultant state tuple (x', y', h')
    """
    r = state[2]
    pos = [state[0], state[1]]
    move = action[0]
    r_change = action[1]
    if 2 <= r <= 4:
        pos[0] += move
    elif 5 <= r <= 7:
        pos[1] += -move
    elif 8 <= r <= 10:
        pos[0] += -move
    else:
        pos[1] += move

    new_r = rotate(r, r_change)
    if check_map_edge(pos):
        new_s = (pos[0], pos[1], new_r)
    else:
        new_s = (state[0], state[1], new_r)
    return new_s


def check_map_edge(new_pos):
    """
    Helper function for new_state that checks whether
    or not movement from current action causes robot
    to go off the grid or not.

    :param new_pos: proposed new position tuple (x, y)
    :return: bool value that is false if robot went off grid
             and true otherwise.
    """
    if new_pos[0] < 0 or new_pos[0] > 7 \
            or new_pos[1] < 0 or new_pos[1] > 7:
        return False
    return True


def rotate(orig_rot, rotate_change):
    """
    Helper function for new_state that handles rotation
    changes. Takes care of edge cases when transitioning
    from 11 to 0 or vice-versa.

    :param orig_rot: original rotation value ranging
                     from 0 to 11
    :param rotate_change: direction of rotation change
                          ranging from -1 to 1
    :return: new rotation value
    """
    if orig_rot < 0 or orig_rot > 11:
        raise ValueError("Rotation value should be between 0 and 11 inclusive.")
    if rotate_change not in [-1, 0, 1]:
        raise ValueError("Rotation change value should be either -1, 0, or 1.")

    new_rotation = orig_rot + rotate_change
    if new_rotation == -1:
        rotation = 11
    elif new_rotation == 12:
        rotation = 0
    else:
        rotation = new_rotation
    return rotation


def init_policy(state_space):
    """
    Function that creates a crude hand-engineered policy.
    Policy prioritizes closing in on the x-position first
    and then the y-position until reaching goal at (5,6).

    :param state_space: the state space of the environment
    :return: a rudimentary policy for all states
    """
    policy = [None]*len(state_space)
    for state in state_space:
        x, y, r = state[0], state[1], state[2]
        x_dist = 5 - x
        y_dist = 6 - y
        move = 1
        rot = 0
        index = si(state)

        # Assign (0, 0) action to goal states.
        if x_dist == 0 and y_dist == 0:
            policy[index] = (0, 0)
            continue

        # Prioritize closing in on x distance first
        # and then y distance movement wise.
        if x_dist != 0:
            dir_x = 1 if x_dist > 0 else -1
            if r in [2, 3, 4]:
                move = dir_x
            elif r in [8, 9, 10]:
                move = -dir_x
        else:
            dir_y = 1 if y_dist < 0 else -1
            if r in [11, 0, 1]:
                move = -dir_y
            elif r in [5, 6, 7]:
                move = dir_y

        # Change rotation depending on whether or not
        # x or y distance is being closed in on.
        if r not in [2, 3, 4, 8, 9, 10] and x_dist != 0:
            if r in [7, 1]:
                rot = 1
            else:
                rot = -1
        elif r not in [11, 0, 1, 5, 6, 7] and y_dist != 0:
            if r in [4, 10]:
                rot = 1
            else:
                rot = -1

        # Make sure no (0, 0) action values are accidentally
        # assigned to states other than the goal.
        assert not (rot == 0 and move == 0)
        policy[index] = (move, rot)

    # Make sure the policy has an action assigned for every state.
    assert None not in policy

    return policy


def display_rewards(reward_space):
    """
    Helper function to graphically view the rewards
    by x, y state using print function. Used to check
    that rewards were properly assigned to the correct
    states.

    :param reward_space: numpy matrix of
                         environment reward space
    """
    # Some axis readjustment due to differences in
    # numpy's matrix and our gridworld origin.
    rs = np.swapaxes(reward_space, 0, 1)
    rs = np.flip(rs, 0)
    print(rs)
