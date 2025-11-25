from abc import ABC, abstractmethod

__all__ = ["Optimizer"]


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass
