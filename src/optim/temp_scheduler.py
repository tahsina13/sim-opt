from abc import ABC, abstractmethod

import numpy as np

from .optimizer import Optimizer

__all__ = [
    "ExponentialTemp",
    "LinearTemp",
    "LogarithmicTemp",
    "TempScheduler",
]


class TempScheduler(ABC):
    optimizer: Optimizer
    step_size: int
    steps: int

    def __init__(self, optimizer: Optimizer, step_size: int = 1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.steps = 0

    @abstractmethod
    def cool(self, temp: float) -> float:
        pass

    def step(self):
        self.steps = 0
        if self.steps % self.step_size == 0 and "temp" in self.optimizer.__dict__:
            self.optimizer.__dict__["temp"] = self.cool(self.optimizer.__dict__["temp"])


class LinearTemp(TempScheduler):
    cooling_rate: float

    def __init__(self, optimizer: Optimizer, cooling_rate: float, step_size: int = 1):
        super().__init__(optimizer, step_size)
        self.cooling_rate = cooling_rate

    def cool(self, temp: float) -> float:
        return max(0, temp - self.cooling_rate)


class ExponentialTemp(TempScheduler):
    cooling_rate: float

    def __init__(self, optimizer: Optimizer, cooling_rate: float, step_size: int = 1):
        super().__init__(optimizer, step_size)
        self.cooling_rate = cooling_rate

    def cool(self, temp: float) -> float:
        return temp * self.cooling_rate

class LogarithmicTemp(TempScheduler):
    def __init__(self, optimizer: Optimizer, step_size: int = 1):
        super().__init__(optimizer, step_size)

    def cool(self, temp: float) -> float:
        return temp / np.log(1 + self.steps)
