import matplotlib.pyplot as plt


class ParkingLot:
    def __init__(self):
        self.ax = self.init_env()
        self.obstacles = self.init_obstacles()

    def init_obstacles(self):
        obstacles = [[250, 0, 100, 800],
                     [1100, 1000, 150, 400],
                     [1350, 0, 150, 400],
                     [750, 600, 400, 150]]
        for i in range(5):
            obstacles.append([750+i*250, 0, 100, 400])
            obstacles.append([750+i*250, 1000, 100, 400])

        # include map boundaries
        obstacles.append([-1, 0, 1, 1400]) # West Boundary
        obstacles.append([2000, 0, 1, 1400]) # East
        obstacles.append([0, -1, 2000, 1]) # South
        obstacles.append([0, 1400, 2000, 1]) # North

        return obstacles

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
