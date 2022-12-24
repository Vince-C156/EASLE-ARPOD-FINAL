import numpy as np


class reward_conditions:

    def __init__(self, chaser):
        self.chaser = chaser
        self.time_limit = 14400
        #self.obstacle_list = obstacle_list
    def inbounds(self): 
        if self.chaser_current_distance() < 1000.0:
           return True
        else:
           return False

    def target_collision(self):
        pass

    def chaser_current_distance(self):
        target_point = self.chaser.docking_point
        curr_pos = self.chaser.state[:3]

        curr_sumsq = np.sum(np.square(target_point - curr_pos))
        curr_dist = np.sqrt(curr_sumsq)

        return curr_dist

    def chaser_last_distance(self):
        target_point = self.chaser.docking_point
        last_pos = self.chaser.state_trace[-2][:3]

        last_sumsq = np.sum(np.square(target_point - last_pos))
        last_dist = np.sqrt(last_sumsq)

        return last_dist

    def is_closer(self):
        """
        if curr_dist < last_dist then curr_dist - last_dist
        represents forward progress
        """
        curr_dist = self.chaser_current_distance()
        last_dist = self.chaser_last_distance()

        if (curr_dist - last_dist) <= 0:
            return True
        else:
            return False

    def in_los(self):
        pass

    def is_velocity_limit(self):
        pass

    def in_time(self):
        """

        """
        if self.chaser.current_step < self.time_limit:
            return True
        return False

    def is_docked(self):
        pass


class reward_formulation(reward_conditions):

    def __init__(self, chaser):
        super().__init__(chaser)

    def terminal_conditions(self):
        """
        check in bounds
        check target collision
        check timelimit

        *check obstacle collision

        return accumulated penality and done status
        """

    def soft_penalities(self):
        """
        check if distance is not increased
        check if is not in los
        check if exceeding velocity limit
        """

    def soft_rewards(self):

