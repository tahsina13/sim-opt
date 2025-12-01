__all__ = ["Optimizer"]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from solvers import StochasticSolver

T = TypeVar("T", bound=StochasticSolver)


class Optimizer(ABC, Generic[T]):
    @abstractmethod
    def step(self):
        pass

    @property
    @abstractmethod
    def solution(self) -> T:
        pass
