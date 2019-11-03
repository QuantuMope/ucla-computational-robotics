import matplotlib.pyplot as plt


class ParkingLot:
    """
    Parking lot environment for RRT Robot.
    """
    def __init__(self):
        self.trajectory_ax = self.init_env()
        self.tree_ax = self.init_env()
        self.obstacles = self.init_obstacles()
        self.goals = self.init_goal_state()

    def get_plots(self):
        return self.trajectory_ax, self.tree_ax

    def init_obstacles(self):
        """
        Initialize list of obstacles where each entry is:
        [lower left corner x,
         lower left corner y,
         width,
         height]

        Padding of 5 mm added as a safety factor against
        possible collisions due to rounding errors when checking
        configuration space.
        """
        obstacles = []
            # [[245, 0, 110, 800],
            #          [1095, 995, 160, 405],
            #          [1345, 0, 160, 405],
            #          [745, 595, 410, 160]]
        # for i in range(5):
        #     obstacles.append([745+i*250, -5, 110, 410])
        #     obstacles.append([745+i*250, 995, 110, 410])

        # include map boundaries
        obstacles.append([-1, 0, 5, 1400]) # West Boundary
        obstacles.append([2000, 0, 5, 1400]) # East
        obstacles.append([0, -1, 2000, 5]) # South
        obstacles.append([0, 1400, 2000, 5]) # North

        return obstacles

    def init_goal_state(self):
        """
        Similar to init_obstacles except for goal state.
        """
        return [1600, 0, 150, 350]

    def init_env(self):
        """
        Plot of the environment.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.grid()
        ax.set_facecolor('gray')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim((0, 2000))
        plt.ylim((0, 1400))

        # # Add wall (yellow).
        # wall = plt.Rectangle((250, 0), 100, 800, color='gold')
        # ax.add_patch(wall)
        #
        # # Add occupied spaces (red).
        # car_spot2 = plt.Rectangle((1100, 1000), 150, 400, color='darkred')
        # car_spot8 = plt.Rectangle((1350, 0), 150, 400, color='darkred')
        # car_block = plt.Rectangle((750, 600), 400, 150, color='darkred')
        # ax.add_patch(car_spot2), ax.add_patch(car_spot8), ax.add_patch(car_block)
        #
        # # Add black lane markers.
        # for i in range(5):
        #     bot_lane_marker = plt.Rectangle((750+i*250, 0), 100, 400, color='w')
        #     top_lane_marker = plt.Rectangle((750+i*250, 1000), 100, 400, color='w')
        #     ax.add_patch(bot_lane_marker), ax.add_patch(top_lane_marker)
        #     plt.text(890+i*250, 170, str(i+6), fontsize=24, fontweight='bold')
        #     plt.text(890+i*250, 1170, str(i+1), fontsize=24, fontweight='bold')

        # Goal state at spot 9.
        goal_spot9 = plt.Rectangle((1605, 0), 145, 350, color='g')
        ax.add_patch(goal_spot9)

        return ax
