__all__ = ["Optimizer"]

from abc import ABC, abstractmethod

from solvers import StochasticSolver


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def solution(self) -> StochasticSolver:
        pass
