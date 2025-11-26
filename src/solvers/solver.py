from abc import ABC, abstractmethod
from typing import Self

import numpy as np

__all__ = ["Solver", "StochasticSolver"]


class Solver(ABC):
    @abstractmethod
    def cost(self) -> float:
        pass

    @abstractmethod
    def __lt__(self, other: Self) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class StochasticSolver(Solver):
    @abstractmethod
    def generate(self, rng: np.random.Generator):
        pass

    @abstractmethod
    def combine(self, other: Self, rng: np.random.Generator):
        pass

    @abstractmethod
    def mutate(self, rng: np.random.Generator):
        pass
