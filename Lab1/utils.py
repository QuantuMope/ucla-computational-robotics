import numpy as np

def si(state):
    """
    Helper function that converts state tuples to
    proper indexes of the state_space.

    :param state: tuple of a state (x, y, r)
    :return: index of the state in state_space
    """
    return state[2] + 12 * state[1] + 96 * state[0]


def ai(action):
    """

    :param action:
    :return: index of action in action_space
    """
    if action == (0, 0):
        return 3
    elif action == (1, -1):
        return 4
    elif action == (1, 0):
        return 5
    elif action == (1, 1):
        return 6
    return (action[1] + 1) + (action[0] + 1) * 3


def get_error_states(state, pe):
    err1_state = new_state(state, (0, 1))
    err2_state = new_state(state, (0, -1))
    return [(state, 1-2*pe),
            (err1_state, pe),
            (err2_state, pe)]


def new_state(state, action):
    """

    :param state: (x, y, r)
    :param action: (move, rotate)
    :return:
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
    if new_pos[0] < 0 or new_pos[0] > 7 \
            or new_pos[1] < 0 or new_pos[1] > 7:
        return False
    return True


def rotate(orig_rot, rotate_change):
    new_rotation = orig_rot + rotate_change
    if new_rotation == -1:
        rotation = 11
    elif new_rotation == 12:
        rotation = 0
    else:
        rotation = new_rotation
    return rotation
