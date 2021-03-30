import numpy as np


class RestrictedBoltzmannMachine:
    def __init__(self, visible_size, hidden_size):
        """
        Implementation of a Restricted Boltzmann machine
        Fields:
            w: connection weights matrix
            b: bias vector
            s: the state
        """
        self.visible_size = visible_size
        self.hidden_size = hidden_size

        self.w = np.random.uniform(-1, 1, size=(self.visible_size, self.hidden_size))
        self.b_v = np.random.uniform(-1, 1, size=self.visible_size)
        self.b_h = np.random.uniform(-1, 1, size=self.hidden_size)
        self.s_v = np.empty(self.visible_size, dtype=np.bool)
        self.s_h = np.empty(self.hidden_size, dtype=np.bool)

    def update_state(self):
        """
        Performs stochastic update of state
        """
        total_inputs_v = self.w @ self.s_h + self.b_v
        total_inputs_h = self.s_v @ self.w + self.b_h
        probabilities_v = 1 / (1 + np.exp(-total_inputs_v))
        probabilities_h = 1 / (1 + np.exp(-total_inputs_h))
        self.s_v = np.random.uniform(0, 1, size=self.visible_size) < probabilities_v
        self.s_h = np.random.uniform(0, 1, size=self.hidden_size) < probabilities_h

    def energy(self):
        return -self.s_h @ self.b_h -self.s_h @ self.b_h - self.s_v @ self.w @ self.s_h

    def gibbs_sampling(self, visible_state=None, iters=10):
        s_h = np.random.uniform(0, 1, self.hidden_size) < 0.5
        if visible_state is None:
            s_v = np.random.uniform(0, 1, self.visible_size) > 0.5
        else:
            s_v = visible_state

        for n in range(iters):
            total_inputs_h = s_v @ self.w + self.b_h
            probabilities_h = 1 / (1 + np.exp(-total_inputs_h))
            s_h = np.random.uniform(0, 1, size=self.hidden_size) < probabilities_h
            if visible_state is None:
                total_inputs_v = self.w @ s_h + self.b_v
                probabilities_v = 1 / (1 + np.exp(-total_inputs_v))
                s_v = np.random.uniform(0, 1, size=self.visible_size) < probabilities_v

        return s_v, s_h

    def update_params_from_gradient(self, dw, db_v, db_h, lr):
        self.w += dw * lr
        self.b_v += db_v * lr
        self.b_h += db_h * lr