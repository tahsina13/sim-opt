__all__ = ["StochasticTSPSolver", "TSPSolver", "TwoOptTSPSolver"]

import sys
from typing import cast

import numpy as np
import numpy.typing as npt

from algos.mst import MST, Edge
from .solver import HeuristicSolver, Solver, StochasticSolver


class TSPSolver(Solver):
    adj_mat: npt.NDArray[np.float64]
    solution: list[int] | None

    def __init__(self, adj_mat: npt.NDArray[np.float64]):
        self.adj_mat = adj_mat

    def __eq__(self, other: object) -> bool:
        other = cast(TSPSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to equate solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost == other.cost

    def __gt__(self, other: Solver) -> bool:
        other = cast(TSPSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to compare solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost < other.cost

    @property
    def cost(self) -> float:
        if self.solution is None:
            return sys.float_info.max
        src = self.solution
        dst = np.roll(self.solution, -1)
        return np.sum(self.adj_mat[src, dst])


class TwoOptTSPSolver(TSPSolver, HeuristicSolver):
    def solve(self):
        # find the MST
        nodes = self.adj_mat.shape[0]
        rows, cols = np.triu_indices(nodes, k=1)
        edges = [Edge(r, c, w) for r, c, w in zip(rows, cols, self.adj_mat[rows, cols])]
        mst = MST(nodes, edges)
        mst_edges = mst()
        adj_list = [[] for _ in range(nodes)]
        for e in mst_edges:
            adj_list[e.u].append(e.v)
            adj_list[e.v].append(e.u)

        # find 2-OPT TSP from MST
        visisted = set()
        stack = [0]
        solution = []
        while stack:
            u = stack.pop()
            if u not in visisted:
                solution.append(u)
            visisted.add(u)
            for v in adj_list[u]:
                if v not in visisted:
                    stack.append(v)
        self.solution = solution


class StochasticTSPSolver(TSPSolver, StochasticSolver):
    def generate(self, rng: np.random.Generator):
        self.solution = rng.permutation(self.adj_mat.shape[0]).tolist()

    def combine(self, other: StochasticSolver, rng: np.random.Generator):
        other = cast(StochasticTSPSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to combine solution '{self.solution}' with '{other.solution}'"
            )
        i = rng.choice(len(self.solution))
        new_solution = self.solution[i:]
        self.solution = new_solution + [
            j for j in other.solution if j not in new_solution
        ]

    def mutate(self, rng: np.random.Generator):
        if self.solution is None:
            raise RuntimeError(f"Failed to mutate solution '{self.solution}'")
        i = rng.choice(len(self.solution))
        j = rng.choice(len(self.solution))
        if i > j:
            i, j = j, i
        self.solution = (
            self.solution[:i] + list(reversed(self.solution[i:j])) + self.solution[j:]
        )
