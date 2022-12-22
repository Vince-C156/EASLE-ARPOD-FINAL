import numpy as np

"""
cw_discrete
chaser_discrete
cw_continous
chaser_continous
"""
class cw_discrete:

    def __init__(self):
        gravitational_constant = float(4041.804)
        a = float(42164000)
        n = np.sqrt(gravitational_constant / (a ** 3))

        self.A = np.array([[(4.0 - 3.0*np.cos(n)), 0, 0, ((1.0/n) * np.sin(n)), ((2/n)*(1-np.cos(n))), 0], 
                          [(6*(np.sin(n) - n)), 1, 0, ((2/n)*(np.cos(n) - 1)), ((1/n)*(4 * np.sin(n) - 3*n)), 0],
                          [0, 0, np.cos(n), 0, 0, ((1/n)*np.sin(n))],
                          [(3*n * np.sin(n)), 0, 0, np.cos(n), (2 * np.sin(n)), 0],
                          [((6*n) * (np.cos(n) - 1)), 0, 0, (-2 * np.sin(n)), (4 * np.cos(n) - 3), 0],
                          [0, 0, (-1 * (n) * np.sin(n)), 0, 0, np.cos(n)]], np.float64)

        self.B = np.array([[( (1 / n**2.0) * (1 - np.cos(n)) ), ( (1 / n**2.0) * ((2.0 * n) - (2.0 * np.sin(n))) ), 0],
                      [( (1 / n ** 2.0) * (2.0 * (np.sin(n) - n)) ), ((-3 / 2) + (4/(n ** 2.0)) * (1 - np.cos(n))), 0],
                      [0, 0, ((1/n**2.0) * (1 - np.cos(n)))],
                      [(np.sin(n) / n ), ((2 / n) * (1 - np.cos(n))), 0],
                      [((2 / n) * (np.cos(n))), (-3 + (4 / n) * np.sin(n)), 0],
                      [0, 0, (np.sin(n) / n)]], np.float64)

    def step(self, state, action, mass):
        """
        input:
             -state:(6,) array
             -action:(3,) array
             -mass: int or float
        """

        x = np.asarray(state, dtype=np.float64)
        u = np.asarray(action, dtype=np.float64)
        u = u / mass

        x = np.reshape(x, (6,1))
        u = np.reshape(u, (3,1))


        x_next = (np.dot(self.A,x)) + ( np.dot(self.B,u) )
        x_next = np.reshape(x_next, (6,))
        return x_next

class chaser_discrete(cw_discrete):

    def __init__(self):
        super().__init__()
        self.state_trace = []
        self.mass = 500 #500kg
        self.current_step = 0
        self.state = self.rand_state()

    def rand_state(self):
        #revise
        pos = np.random.randint(low=-1000, high=1000, size=3)
        vel = np.random.randint(low=-10, high=10, size=3)

        state = np.concatenate((pos, vel), axis=None, dtype=np.float64)
        print("generated state")
        print(state)

        return state

    def get_next(self, action):
        next_x = super().step(self.state, action, self.mass)
        return next_x

    def update_state(self, state):
        self.state_trace.append(state)
        self.state = state
        self.current_step += 1
        print(f"state {self.state}, step {self.current_step}")

    def get_state_trace(self):
        return self.state_trace

    def reset(self):
        self.state_trace = []
        self.mass = 500 #500kg
        self.current_step = 0
        self.state = self.rand_state()
        print("reset to default")


class cw_continous:
    pass

class chaser_continous:
    pass
