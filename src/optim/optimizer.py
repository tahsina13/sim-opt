__all__ = ["Optimizer"]

from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass
