import torch
import numpy as np

class LogBarrierLoss():
    def __init__(self, t):
        self.t = t

    def penalty(self, z):
        if z <= - 1 / self.t**2:
            return - torch.log(-z) / self.t
        else:
            return self.t * z - np.log(1 / (self.t**2)) / self.t + 1 / self.t
