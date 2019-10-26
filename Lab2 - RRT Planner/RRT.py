import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from ParkingLot import ParkingLot
from utils import rpm_to_rps, add_angles


# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100




class Robot:
    def __init__(self, start_x, start_y, start_theta, parking_plot):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_rad = WHEEL_RADIUS

        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        self.ax = parking_plot
        # br - bottom right, bl - ...
        self.br = (self.x+45, self.y-75)
        self.bl = (self.x-45, self.y-75)
        self.tr = (self.x+45, self.y+25)
        self.tl = (self.x-45, self.y+25)
        self.frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length, facecolor='cyan', linewidth=1, edgecolor='magenta')
        self.ax.plot(self.x, self.y, marker='o', color='blue')
        self.ax.add_patch(self.frame)

    def drive(self, left_rpm, right_rpm):
        left_vel = rpm_to_rps(left_rpm)
        right_vel = rpm_to_rps(right_rpm)
        central_vel = (left_vel + right_vel) / 2
        dtheta = -(self.wheel_rad/self.width) * (right_vel - left_vel)
        self.theta = add_angles(self.theta, dtheta)
        dx = central_vel * np.sin(self.theta*2*np.pi/360)
        dy = central_vel * np.cos(self.theta*2*np.pi/360)

        self.x += dx
        self.y += dy
        self.frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length,
                              facecolor='cyan', linewidth=1, edgecolor='magenta')
        ts = self.ax.transData
        tr = matplotlib.transforms.Affine2D().rotate_deg_around(self.x, self.y, -self.theta)
        t = tr + ts
        self.frame.set_transform(t)
        self.ax.add_patch(self.frame)
        self.ax.plot(self.x, self.y, marker='o', color='blue')

    def calculate_config_space(self, obstacles):
        config_space = []
        config_frame = plt.Rectangle((self.x-45, self.y-75), self.width, self.length)
        ts = self.ax.transData
        for theta in range(0, 361):
            tr = matplotlib.transforms.Affine2D().rotate_deg_around(self.x, self.y, -theta)
            t = tr + ts
            config_frame.set_transform(t)
            x, y = config_frame.get_x(), config_frame.get_y()

            # bottom left, top left, top right, bottom right
            corners = np.array([[x, y],
                                [x, y+self.length],
                                [x+self.width, y+self.length],
                                [x+self.width, y]])
            corners = tr.transform(corners)
            corner_to_center = (corners - np.array([self.x, self.y])) * -1
            obstacle_boundaries = []
            for obstacle in obstacles:
                boundary = []
                bottom = obstacle[1]
                top = bottom + obstacle[3]
                left = obstacle[0]
                right = left + obstacle[2]
                a, b, c, d = 0, 1, 2, 3
                if theta < 180:
                    a, b, c, d = 3, 0, 1, 2
                boundary.append((left + corner_to_center[a][0], top + corner_to_center[a][1], theta))
                boundary.append((right + corner_to_center[a][0], top + corner_to_center[a][1], theta))
                boundary.append((right + corner_to_center[b][0], top + corner_to_center[b][1], theta))
                boundary.append((right + corner_to_center[b][0], bottom + corner_to_center[b][1], theta))
                boundary.append((right + corner_to_center[c][0], bottom + corner_to_center[c][1], theta))
                boundary.append((left + corner_to_center[c][0], bottom + corner_to_center[c][1], theta))
                boundary.append((left + corner_to_center[d][0], bottom + corner_to_center[d][1], theta))
                boundary.append((left + corner_to_center[d][0], top + corner_to_center[d][1], theta))
                obstacle_boundaries.append(boundary)
            config_space.append(obstacle_boundaries)

        fig = plt.figure(2)
        ax = Axes3D(fig)
        ax.set_xlim3d((0, 2000))
        ax.set_ylim3d((0, 1400))
        ax.set_zlim3d((0, 360))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Theta")
        plt.title("Configuration Space")
        for theta, all_obstacles in enumerate(config_space):
            color = 'steelblue'
            if theta % 20 == 0: color = 'black'
            for obstacle_verts in all_obstacles:
                ax.add_collection3d(Poly3DCollection([obstacle_verts], edgecolors=color))


    # Question 2(a)
    def find_nearest(self):




def main():
    env = ParkingLot()
    robot = Robot(500, 1000, 0, env.ax)
    robot.calculate_config_space(env.obstacles)

    plt.show()



if __name__ == "__main__":
    main()
