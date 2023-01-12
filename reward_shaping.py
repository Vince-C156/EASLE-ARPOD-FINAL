import numpy as np
import pdb


class reward_conditions:

    def __init__(self, chaser):
        self.chaser = chaser
        self.time_limit = 14400
        #self.obstacle_list = obstacle_list
    def inbounds(self): 
        if self.chaser_current_distance() < 1500.0:
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
        if len(self.chaser.state_trace) < 2:
            return np.linalg.norm(self.chaser.state[:3] - target_point)
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
        los_len = 800.0
        dock_pos = np.array(self.chaser.docking_point, copy=True)
        theta = self.chaser.theta_cone

        p = np.array(self.chaser.state[:3], dtype=np.float64, copy=True) - dock_pos

        #c_hat = dock_pos / np.linalg.norm(dock_pos)
        #c = c_hat * los_len
        c = dock_pos + np.array([0.0, 800, 0.0], dtype=np.float64)

        condition = np.cos(theta / 2.0)
        #value = np.dot(p,c) / ( np.linalg.norm(p) * np.linalg.norm(c) )
        value = np.dot(p,c) / (np.absolute(p) @ np.absolute(c))
        if value >= condition:
            return True
        return False

    def is_velocity_limit(self):
        state = np.array(self.chaser.state, copy=True)
        #print(f'testing state {self.chaser.state}')
        vel = state[3:]

        distance = self.chaser_current_distance()
        if distance < self.chaser.phase3_d:
            if np.linalg.norm(vel) > 0.05:
                 return False
        elif distance < self.chaser.slowzone_d:
            if np.linalg.norm(vel) > 8.0:
                print(f'failed chaser velocity {np.linalg.norm(vel)}')
                return False
        return True

    def in_phase3(self):
        dist = self.chaser_current_distance()
        if dist <= self.chaser.phase3_d:
            return True
        return False

    def in_slowzone(self):
        dist = self.chaser_current_distance()
        if dist <= self.chaser.slowzone_d and dist >= self.chaser.phase3_d:
            return True
        return False

    def in_time(self):
        """

        """
        if self.chaser.current_step < self.time_limit:
            return True
        return False

    def l2norm_constraint(self):
        chaser_p = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
        chaser_v = np.array(self.chaser.state[3:], dtype=np.float64, copy=True)
        pos = np.square(self.chaser.docking_point - chaser_p, dtype=np.float64)
        vel = np.square(chaser_v)
        print(f'chaser_p {chaser_p}')
        print(f'difference {pos}')
        print(f'vel {vel}')
        state = np.concatenate((pos,vel))
        print(f'state {state}')
        l2_norm = np.sqrt(np.sum(state))
        print(f'l2 norm {l2_norm}')
        if l2_norm < 0.001:
            return 2
        elif l2_norm < 0.005:
            return 1
        else:
            return 0


    def is_docked(self):
        pos = np.array(self.chaser.state[:3], dtype=np.float64, copy=True)
        vel = np.array(self.chaser.state[3:], dtype=np.float64, copy=True)
        if self.in_phase3() and self.in_los():
            pos_cm = (pos - self.chaser.docking_point) * 100.0
            vel_cm = vel * 100.0
            #print(f'position in centimeters {pos_cm}')
            #print(f'velocity in centimeters {vel_cm}')
            #if less than 20cm away and less than 1cm/s
            if np.linalg.norm(pos_cm) < 20 and np.linalg.norm(vel_cm) < 1:
                return True
        return False

    def docking_collision(self):
        pass

class reward_formulation(reward_conditions):

    def __init__(self, chaser):
        super().__init__(chaser)
        self.time_inlos = 0
        self.time_slowzone = 0
        self.time_inlos_slowzone = 0
        self.time_inlos_phase3 = 0

    def terminal_conditions(self):
        """
        check in bounds
        check target collision
        check timelimit

        *check obstacle collision

        return accumulated penality and done status
        """
        if not super().in_time():
            print('mission time limit exceeded')
            penality = -1
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
        if not super().in_los():
            penality += -1.0
        else:
            self.time_inlos += 1
        return penality
 

    def soft_rewards(self):
        """
        check if closer
        check if los
        """
        reward = 0
        if super().l2norm_constraint() == 2:
            reward += 2

        if super().l2norm_constraint() == 1:
            reward += 1

        return reward

    def win_conditions(self):
        if super().is_docked():
            reward = 5
            done = True
            return reward, done
        return 0, False

    def reset_counts(self):
        self.time_inlos = 0
        self.time_slowzone = 0
        self.time_inlos_slowzone = 0
        self.time_inlos_phase3 = 0

