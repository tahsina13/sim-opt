__all__ = ["SimulatedAnnealing"]

import numpy as np

from solvers import StochasticSolver
from .optimizer import Optimizer

RNGS = ("selection", "combination", "mutation")


class SimulatedAnnealing(Optimizer):
    rngs: dict[str, np.random.Generator]
    temp: float
    solver: StochasticSolver

    def __init__(
        self,
        solver: StochasticSolver,
        temp: float,
        rngs: dict[str, np.random.Generator] | None = None,
    ):
        if rngs is None:
            rngs = {}
        for name in RNGS:
            if name not in rngs:
                rngs[name] = np.random.default_rng()
        self.rngs = rngs
        self.temp = temp
        self.solver = solver

    def step(self):
        new_solver = StochasticSolver.combined(
            self.solver, self.solver, self.rngs["combination"]
        )
        new_solver.mutate(self.rngs["mutation"])
        if self.solver > new_solver:
            energy = abs(self.solver.cost() - new_solver.cost())
            prob = np.exp(-energy / self.temp)
        else:
            prob = 1.0
        if self.rngs["selection"].uniform() <= prob:
            self.solver = new_solver
