import numpy as np


class BoltzmannMachine:
    def __init__(self, visible_size, hidden_size):
        """
        Implementation of a (non-restricted) Boltzmann machine
        Fields:
            w: connection weights matrix
            b: bias vector
            s: the state
        """
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.total_size = visible_size + hidden_size

        self.w = np.random.uniform(-1, 1, size=(self.total_size, self.total_size))
        self.b = np.random.uniform(-1, 1, size=self.total_size)
        self.s = np.empty(self.total_size, dtype=np.bool)

    def update_state(self):
        """
        Performs stochastic update of state
        """
        total_inputs = self.w @ self.s + self.b
        probabilities = 1 / (1 + np.exp(-total_inputs))
        self.s = np.random.uniform(0, 1, size=self.total_size) < probabilities

    def energy(self):
        return -self.s @ self.b - self.s @ self.w @ self.s

    def gibbs_sampling(self, visible_state=None, iters=10):
        if visible_state is not None:
            s = np.concatenate((visible_state, np.random.uniform(0, 1, self.hidden_size) < 0.5))
        else:
            s = np.random.uniform(0, 1, self.total_size) > 0.5

        if visible_state is not None:
            for n in range(iters):
                for i in range(self.visible_size, self.total_size):
                    total_input = self.w[i] @ self.s + self.b[i]
                    probability = 1 / (1 + np.exp(-total_input))
                    s[i] = np.random.uniform(0, 1) < probability
        else:
            for n in range(iters):
                for i in range(self.total_size):
                    total_input = self.w[i] @ self.s + self.b[i]
                    probability = 1 / (1 + np.exp(-total_input))
                    s[i] = np.random.uniform(0, 1) < probability

        return s

    def update_params_from_gradient(self, dw, db, lr):
        self.w += dw * lr
        self.b += db * lr