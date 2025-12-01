__all__ = ["SimulatedAnnealing"]

from typing import TypeVar, cast

import numpy as np
from solvers import StochasticSolver

from .optimizer import Optimizer
from .temp_scheduler import TempScheduler

T = TypeVar("T", bound=StochasticSolver)

RNGS = ("selection", "combination", "mutation")


class SimulatedAnnealing(Optimizer[T]):
    solver: T
    temp_sched: TempScheduler
    rngs: dict[str, np.random.Generator]
    prob: float

    def __init__(
        self,
        solver: T,
        temp_sched: TempScheduler,
        seed_seq: np.random.SeedSequence | None = None,
    ):
        self.solver = solver
        self.temp_sched = temp_sched
        if not seed_seq:
            seed_seq = np.random.SeedSequence()
        seeds = seed_seq.spawn(len(RNGS))
        self.rngs = {}
        for name, seed in zip(RNGS, seeds):
            self.rngs[name] = np.random.default_rng(seed)
        self.prob = 0.5

    def step(self):
        # perturbe current solution
        new_solver = StochasticSolver.combined(
            self.solver, self.solver, self.rngs["combination"]
        )
        new_solver.mutate(self.rngs["mutation"])

        # accept/reject solution
        if self.solver > new_solver:
            energy = abs(self.solver.cost - new_solver.cost)
            self.prob = np.exp(-energy / self.temp_sched.temp)
        else:
            self.prob = 1.0
        if self.rngs["selection"].uniform() <= self.prob:
            self.solver = cast(T, new_solver)

    @property
    def solution(self) -> T:
        return self.solver
