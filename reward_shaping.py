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
        los_len = 1000.0
        dock_pos = self.chaser.docking_point
        theta = self.chaser.theta_cone

        p = self.chaser.state[:3] - dock_pos

        c_hat = dock_pos / np.linalg.norm(dock_pos)
        c = c_hat * los_len

        condition = np.cos(theta / 2.0)
        value = np.dot(p,c) / ( np.linalg.norm(p) * np.linalg.norm(c) )

        if condition <= value:
            return True
        return False

    def is_velocity_limit(self):
        state = self.chaser.state
        print(f'testing state {self.chaser.state}')
        vel = state[3:]

        distance = self.chaser_current_distance()
        if distance < self.chaser.phase3_d:
            if np.linalg.norm(vel) > 0.05:
                 return False
        elif distance < self.chaser.slowzone_d:
            if np.linalg.norm(vel) > 0.2:
                return False
        return True

    def in_phase3(self):
        dist = self.chaser_current_distance()
        if dist <= self.chaser.phase3_d:
            return True
        return False

    def in_time(self):
        """

        """
        if self.chaser.current_step < self.time_limit:
            return True
        return False

    def is_docked(self):
        pos = self.chaser.state[:3]
        vel = self.chaser.state[3:]
        if self.in_phase3() and self.in_los():
            pos_cm = (pos - self.chaser.docking_point) / 100.0
            vel_cm = vel / 100.0
            print(f'position in centimeters {pos_cm}')
            print(f'velocity in centimeters {vel_cm}')
            #if less than 20cm away and less than 1cm/s
            if np.linalg.norm(pos_cm) < 20 and np.linalg.norm(vel_cm) < 1:
                return True
        return False

    def docking_collision(self):
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
        if not super().inbounds():
            penality = -20
            done = True
            return penality, done
        if not super().in_time():
            penality = -20
            done = True
            return penality, done
        if super().in_phase3 and not super().in_los():
            penality = -10
            done = True
            return penality, done
        return 0, False
    def soft_penalities(self):
        """
        check if distance is not increased
        check if is not in los
        check if exceeding velocity limit
        """
        penality = 0
        if not super().is_closer():
            penality -= 1
        if not super().is_velocity_limit():
            penality -= 2
        return penality

    def soft_rewards(self):
        """
        check if closer
        check if los
        """
        reward = 0
        if super().is_closer():
            reward += 2
        if super().in_los():
            reward += 3
        return reward

    def win_conditons(self):
        if super().is_docked():
            reward = 50
            done = True
            return reward, done
         return 0, False
