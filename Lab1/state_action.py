class State:
    def __init__(self, x, y, rotation):
        self.x = x
        self.y = y
        self.rotation = rotation

    def rotate(self, rotate_change):
        new_rotation = self.rotation + rotate_change
        if new_rotation == -1:
            self.rotation = 11
        elif new_rotation == 12:
            self.rotation = 0
        else:
            self.rotation = new_rotation

    def copy(self):
        return State(self.x, self.y, self.rotation)

    def tuple(self):
        return self.x, self.y, self.rotation


class Action:
    def __init__(self, move, rotate):
        if move < -1 or move > 1 or rotate < -1 or rotate > 1:
            raise ValueError("Improper move or rotate value input")
        self.move = move
        self.rotate = rotate
