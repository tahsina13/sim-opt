__all__ = ["GeneticAlgorithm"]

from typing import Sequence

import numpy as np
from solvers import StochasticSolver

from .optimizer import Optimizer
from .temp_scheduler import TempScheduler

RNGS = ("selection", "combination", "mutation")


class GeneticAlgorithm(Optimizer):
    solvers: Sequence[StochasticSolver]
    temp_sched: TempScheduler
    rngs: dict[str, np.random.Generator]
    probs: list[float]

    def __init__(
        self,
        solvers: Sequence[StochasticSolver],
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
        # compute fitness using softmax
        best_sol = max(self.solvers)
        energies = np.asarray([best_sol.cost - s.cost for s in self.solvers])
        exp_energies = np.exp(energies / self.temp_sched.temp)
        self.probs = exp_energies / exp_energies.sum()

        # get new generation
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
        for i in range(len(self.solvers)):
            self.solvers[i].__dict__.update(new_solvers[i].__dict__)
