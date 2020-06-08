import numpy as np


class RRT_Node:
    """
    Node class for a RRT tree.

    :param state: state of the node, tuple (x, y, theta)
    """
    def __init__(self, state):

        self.x = state[0]
        self.y = state[1]
        self.theta = state[2]

        self.parent = None
        self.actions = []
        self.path = []
        self.cost = 0

    def get_state(self):
        """
        Get function for the robot pose stored in the node.

        :return: numpy array of node x, y, and theta
        """
        return np.array([self.x, self.y, self.theta])

    def set_parent(self, parent_node, actions, path):
        """
        Sets the parent of the node. Updates the sequence
        of actions and states from parent to caller node.
        Updates cost which is equal to the number of actions
        per time step it took to get to caller node from root.

        :param parent_node: Node from which the path starts
        :param actions: sequence of actions in the trajectory
        :param path: sequence of states in the trajectory
        """
        self.parent = parent_node
        self.actions = actions
        self.path = path
        self.cost = self.parent.cost + len(self.actions)

    def equals(self, other_node):
        """
        Checks to see if other_node is the same as
        the caller node.

        :param other_node: Node to be compared with.
        :return: True if node is the same. False otherwise.
        """
        return np.array_equal(self.get_state(), other_node.get_state())
