import numpy as np

class LaplaceNoise:
    def __init__(self, epsilon=1.0, sensitivity=1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def add_noise(self, value):
        noise = np.random.laplace(0, self.scale)
        return value + noise