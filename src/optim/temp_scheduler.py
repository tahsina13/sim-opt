__all__ = [
    "ExponentialTemp",
    "LinearTemp",
    "LogarithmicTemp",
    "TempScheduler",
]

from abc import ABC, abstractmethod

import numpy as np


class TempScheduler(ABC):
    temp: float

    def __init__(self, temp: float):
        self.temp = temp

    @abstractmethod
    def step(self):
        pass


class LinearTemp(TempScheduler):
    cooling_rate: float

    def __init__(self, temp: float, cooling_rate: float):
        super().__init__(temp)
        self.cooling_rate = cooling_rate

    def step(self):
        self.temp = max(0, self.temp - self.cooling_rate)


class ExponentialTemp(TempScheduler):
    cooling_rate: float

    def __init__(self, temp: float, cooling_rate: float):
        super().__init__(temp)
        self.cooling_rate = cooling_rate

    def step(self):
        self.temp *= self.cooling_rate


class LogarithmicTemp(TempScheduler):
    steps: int

    def __init__(self, temp: float):
        super().__init__(temp)
        self.steps = 0

    def step(self):
        self.steps += 1
        self.temp = self.temp / np.log(1 + self.steps)

    def cool(self, temp: float) -> float:
        return temp / np.log(1 + self.steps)
