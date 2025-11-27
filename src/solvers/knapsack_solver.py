__all__ = ["KnapsackSolver", "StochasticKnapsackSolver"]

import sys
from typing import cast

import numpy as np
import numpy.typing as npt

from .solver import Solver, StochasticSolver


class KnapsackSolver(Solver):
    weights: npt.NDArray[np.int_]
    limit: int
    solution: list[bool] | None

    def __init__(self, weights: npt.NDArray[np.int_], limit: int):
        self.weights = weights
        self.limit = limit

    def cost(self) -> float:
        if self.solution is None:
            return sys.float_info.min
        total_weight = np.sum(self.weights[np.flatnonzero(self.solution)])
        if total_weight > self.limit:
            return 0
        return float(total_weight)

    def __gt__(self, other: Solver) -> bool:
        other = cast(KnapsackSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to compare solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost() > other.cost()

    def __eq__(self, other: object) -> bool:
        other = cast(KnapsackSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to equate solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost() == other.cost()


class StochasticKnapsackSolver(KnapsackSolver, StochasticSolver):
    def generate(self, rng: np.random.Generator):
        self.solution = rng.integers(0, 2, len(self.weights)).tolist()

    def combine(self, other: StochasticSolver, rng: np.random.Generator):
        other = cast(StochasticKnapsackSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to combine solution '{self.solution}' with '{other.solution}'"
            )
        i = rng.choice(len(self.solution))
        self.solution = self.solution[i:] + self.solution[:i]

    def mutate(self, rng: np.random.Generator):
        if self.solution is None:
            raise RuntimeError(f"Failed to mutate solution '{self.solution}'")
        i = rng.choice(len(self.solution))
        self.solution[i] = not self.solution[i]
