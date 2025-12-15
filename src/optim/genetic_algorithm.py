__all__ = ["GeneticAlgorithm"]

from typing import Sequence, TypeVar

import numpy as np
from solvers import StochasticSolver

from .optimizer import Optimizer
from .temp_scheduler import TempScheduler

T = TypeVar("T", bound=StochasticSolver)

RNGS = ("selection", "combination", "mutation")


class GeneticAlgorithm(Optimizer[T]):
    solvers: Sequence[T]
    temp_sched: TempScheduler
    rngs: dict[str, np.random.Generator]
    probs: list[float]

    def __init__(
        self,
        solvers: Sequence[T],
        temp_sched: TempScheduler,
        seed_seq: np.random.SeedSequence | None = None,
    ):
        self.solvers = solvers
        self.temp_sched = temp_sched
        if not seed_seq:
            seed_seq = np.random.SeedSequence()
        seeds = seed_seq.spawn(len(RNGS))
        self.rngs = {}
        for name, seed in zip(RNGS, seeds):
            self.rngs[name] = np.random.default_rng(seed)
        self.probs = [1.0 / len(solvers) for _ in solvers]

    def step(self):
        costs = np.asarray([s.cost for s in self.solvers])
        self.diversity = np.std(costs)
        
        # Figures out if it's max or min and assigns accordingly
        best_sol = max(self.solvers)
        is_maximization = (best_sol.cost == np.max(costs))
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        
        if is_maximization:
            fitness_values = costs - min_cost + 1.0
        else:
            fitness_values = max_cost - costs + 1.0
        exponent = fitness_values / max(self.temp_sched.temp, 1e-10)
        exponent = exponent - np.max(exponent)  
        exp_fitness = np.exp(exponent)
        self.probs = exp_fitness / exp_fitness.sum()
        
        new_solvers = []
        for _ in range(len(self.solvers) // 2):
            i = self.rngs["selection"].choice(len(self.solvers), p=self.probs)
            j = self.rngs["selection"].choice(len(self.solvers), p=self.probs)
            a = StochasticSolver.combined(
                self.solvers[i], self.solvers[j], self.rngs["combination"]
            )
            b = StochasticSolver.combined(
                self.solvers[j], self.solvers[i], self.rngs["combination"]
            )
            a.mutate(self.rngs["mutation"])
            b.mutate(self.rngs["mutation"])
            new_solvers.append(a)
            new_solvers.append(b)
        self.solvers = new_solvers

    @property
    def solution(self) -> T:
        return max(self.solvers)
