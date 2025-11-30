__all__ = ["SimulatedAnnealing"]

import numpy as np

from solvers import StochasticSolver
from .optimizer import Optimizer
from .temp_scheduler import TempScheduler

RNGS = ("selection", "combination", "mutation")


class SimulatedAnnealing(Optimizer):
    solver: StochasticSolver
    temp_sched: TempScheduler
    rngs: dict[str, np.random.Generator]
    prob: float

    def __init__(
        self,
        solver: StochasticSolver,
        temp_sched: TempScheduler,
        rngs: dict[str, np.random.Generator] | None = None,
    ):
        self.solver = solver
        self.temp_sched = temp_sched
        if rngs is None:
            rngs = {}
        for name in RNGS:
            if name not in rngs:
                rngs[name] = np.random.default_rng()
        self.rngs = rngs
        self.prob = 1.0

    def step(self):
        new_solver = StochasticSolver.combined(
            self.solver, self.solver, self.rngs["combination"]
        )
        new_solver.mutate(self.rngs["mutation"])
        if self.solver > new_solver:
            energy = abs(self.solver.cost - new_solver.cost)
            self.prob = np.exp(-energy / self.temp_sched.temp)
        else:
            self.prob = 1.0
        if self.rngs["selection"].uniform() <= self.prob:
            self.solver.__dict__.update(new_solver.__dict__)
