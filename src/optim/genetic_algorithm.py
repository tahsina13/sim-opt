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
        ranked_solvers = sorted(enumerate(self.solvers), key=lambda x: x[1], reverse=True) 
        ranks = np.arange(len(self.solvers), 0, -1)  
        energies = ranks.astype(float)
        exp_energies = np.exp(energies / self.temp_sched.temp)
        ranked_probs = exp_energies / exp_energies.sum()
        self.probs = [0.0] * len(self.solvers)
        for prob, (orig_idx, _) in zip(ranked_probs, ranked_solvers):
            self.probs[orig_idx] = prob
        
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
