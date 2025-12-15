__all__ = ["KnapsackSolver", "StochasticKnapsackSolver"]

import sys
from typing import cast

import numpy as np
import numpy.typing as npt

from .solver import Solver, StochasticSolver


class KnapsackSolver(Solver):
    weights: npt.NDArray[np.int64]
    values: npt.NDArray[np.int64]
    limit: int
    solution: list[bool] | None

    def __init__(self, weights: npt.NDArray[np.int_], values: npt.NDArray[np.int_], limit: int):
        self.weights = weights
        self.values = values
        self.limit = limit

    def __eq__(self, other: object) -> bool:
        other = cast(KnapsackSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to equate solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost == other.cost

    def __gt__(self, other: Solver) -> bool:
        other = cast(KnapsackSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to compare solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost > other.cost

    @property
    def cost(self) -> float:
        if self.solution is None:
            return 0.0
        total_weight = np.sum(self.weights[np.flatnonzero(self.solution)])
        if total_weight > self.limit:
            over = total_weight - self.limit
            return -float(over)
        return float(np.sum(self.values[np.flatnonzero(self.solution)]))


class StochasticKnapsackSolver(KnapsackSolver, StochasticSolver):
    def generate(self, rng: np.random.Generator):
        self.solution = [False] * len(self.weights)
        current_weight = 0
        
        order = rng.permutation(len(self.weights))
        
        for idx in order:
            if rng.random() < 0.5 and current_weight + self.weights[idx] <= self.limit:
                self.solution[idx] = True
                current_weight += self.weights[idx]

    def combine(self, other: StochasticSolver, rng: np.random.Generator):
        other = cast(StochasticKnapsackSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to combine solution '{self.solution}' with '{other.solution}'"
            )
        
        parent1_items = [i for i, b in enumerate(self.solution) if b]
        parent2_items = [i for i, b in enumerate(other.solution) if b]
        
        # start with common items and add unique items until limit is reached
        common_items = set(parent1_items) & set(parent2_items)
        self.solution = [i in common_items for i in range(len(self.solution))]
        current_weight = np.sum(self.weights[list(common_items)])
        unique_items = (set(parent1_items) | set(parent2_items)) - common_items
        unique_items = list(unique_items)
        rng.shuffle(unique_items)
        
        for idx in unique_items:
            if current_weight + self.weights[idx] <= self.limit:
                self.solution[idx] = True
                current_weight += self.weights[idx]
        
        all_items = set(range(len(self.weights)))
        new_items = all_items - set(parent1_items) - set(parent2_items)
        new_items = list(new_items)
        rng.shuffle(new_items)
        
        for idx in new_items:
            if current_weight + self.weights[idx] <= self.limit:
                # 10% chance to add new item
                if rng.random() < 0.1:  
                    self.solution[idx] = True
                    current_weight += self.weights[idx]

    def mutate(self, rng: np.random.Generator):
        if self.solution is None:
            raise RuntimeError(f"Failed to mutate solution '{self.solution}'")
        
        selected = [i for i, b in enumerate(self.solution) if b]
        unselected = [i for i, b in enumerate(self.solution) if not b]
        
        if not selected or not unselected:
            return
        
        # swap mutation
        remove_idx = rng.choice(selected)
        add_idx = rng.choice(unselected)
        
        current_weight = np.sum(self.weights[np.flatnonzero(self.solution)])
        new_weight = current_weight - self.weights[remove_idx] + self.weights[add_idx]
        
        if new_weight <= self.limit:
            self.solution[remove_idx] = False
            self.solution[add_idx] = True
        else:
            self.solution[remove_idx] = False