import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# in mm.
WHEEL_RADIUS = 25
ROBOT_WIDTH = 90
ROBOT_LENGTH = 100

def RPM_to_vel(rpm):
    if rpm > 60 or rpm < -60:
        raise ValueError("Invalid rpm. Range: -60 to 60")
    tire_circum = 2*np.pi*WHEEL_RADIUS
    rps = rpm/60
    distance_per_sec = np.round(tire_circum * rps)
    return distance_per_sec

class Robot:
    def __init__(self, start_x, start_y, start_theta, parking_plot):
        self.width = ROBOT_WIDTH
        self.length = ROBOT_LENGTH
        self.wheel_rad = WHEEL_RADIUS

        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        self.ax = parking_plot

    def update_pos(self):
        # work on transform
        ts = self.ax.transData
        tr = matplotlib.transforms.Affine2D().rotate_deg_around(self.x, self.y, self.theta)
        t = tr + ts
        pos = plt.Rectangle((self.x-45, self.y-75), self.width, self.length, color='magenta', transform=t)
        self.ax.plot(self.x, self.y, marker='o', color='blue')
        self.ax.add_patch(pos)

    def drive(self, left_rpm, right_rpm):
        left_vel = RPM_to_vel(left_rpm)
        right_vel = RPM_to_vel(right_rpm)
        central_vel = (left_vel + right_vel) / 2
        dtheta = (self.wheel_rad/self.width) * (right_vel - left_vel)
        dx = central_vel * np.sin(dtheta)
        dy = central_vel * np.cos(dtheta)
        self.x += dx
        self.y += dy
        self.theta += np.degrees(dtheta)

class ParkingLot:
    def __init__(self):
        self.ax = self.init_env()

    def init_env(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.grid()
        ax.set_facecolor('gray')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((0, 2000))
        plt.ylim((0, 1400))

        # Add wall (yellow).
        wall = plt.Rectangle((250, 0), 100, 800, color='gold')
        ax.add_patch(wall)

        # Add occupied spaces (red).
        car_spot2 = plt.Rectangle((1100, 1000), 150, 400, color='darkred')
        car_spot8 = plt.Rectangle((1350, 0), 150, 400, color='darkred')
        car_block = plt.Rectangle((750, 600), 400, 150, color='darkred')
        ax.add_patch(car_spot2), ax.add_patch(car_spot8), ax.add_patch(car_block)

        # Add black lane markers.
        for i in range(5):
            bot_lane_marker = plt.Rectangle((750+i*250, 0), 100, 400, color='w')
            top_lane_marker = plt.Rectangle((750+i*250, 1000), 100, 400, color='w')
            ax.add_patch(bot_lane_marker), ax.add_patch(top_lane_marker)
            plt.text(890+i*250, 170, str(i+6), fontsize=24, fontweight='bold')
            plt.text(890+i*250, 1170, str(i+1), fontsize=24, fontweight='bold')

        # Goal state at spot 9.
        goal_spot9 = plt.Rectangle((1600, 0), 150, 200, color='g')
        ax.add_patch(goal_spot9)

        return ax

def main():
    env = ParkingLot()
    robot = Robot(500, 1000, 0, env.ax)
    robot.update_pos()
    robot.drive(60, -60)
    robot.update_pos()
    robot.drive(60, -60)
    robot.update_pos()

    plt.show()



if __name__ == "__main__":
    main()
