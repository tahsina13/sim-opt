from .optimizer import Optimizer

__all__ = ["SimulatedAnnealing"]


class SimulatedAnnealing(Optimizer):
    temp: float

    def __init__(self, temp: float):
        self.temp = temp

    def step(self):
        pass
