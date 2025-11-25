from abc import ABC, abstractmethod

from .optimizer import Optimizer

__all__ = [
    "LinearTemp",
    "TempScheduler",
]


class TempScheduler(ABC):
    optimizer: Optimizer

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def step(self):
        pass


class LinearTemp(TempScheduler):
    step_size: int
    steps: int

    def __init__(self, optimizer: Optimizer, step_size: int = 1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.steps = 0

    def step(self):
        self.steps += 1
        if self.steps % self.step_size == 0:
            if "temp" in self.optimizer.__dict__:
                self.optimizer.__dict__["temp"] -= 1
