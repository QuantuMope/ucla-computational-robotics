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

def init_policy(state_space):
    policy = [None]*len(state_space)
    for state in state_space:
        x, y, r = state[0], state[1], state[2]
        x_dist = 5 - x
        y_dist = 6 - y
        move = 1
        rot = 0
        index = si(state)

        if x_dist == 0 and y_dist == 0:
            policy[index] = (0, 0)
            continue

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

        assert not (rot == 0 and move == 0)
        policy[index] = (move, rot)

    assert None not in policy
    return policy
