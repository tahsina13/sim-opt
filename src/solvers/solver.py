__all__ = ["Solver", "StochasticSolver"]

import copy
from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Self

import numpy as np


@total_ordering
class Solver(ABC):
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __gt__(self, other: Self) -> bool:
        pass

    @property
    @abstractmethod
    def cost(self) -> float:
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

    @classmethod
    def combined(cls, this: Self, other: Self, rng: np.random.Generator):
        this = copy.copy(this)
        this.combine(other, rng)
        return this

    @classmethod
    def mutated(cls, this: Self, rng: np.random.Generator):
        this = copy.copy(this)
        this.mutate(rng)
        return this
