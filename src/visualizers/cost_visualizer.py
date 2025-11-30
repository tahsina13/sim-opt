__all__ = ["CostVisualizer"]

from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from solvers import Solver

from .visualizer import Visualizer


class CostVisualizer(Visualizer):
    ax: Axes
    niter: int
    solver: Solver | Sequence[Solver]
    line: Line2D | None

    def __init__(self, ax: Axes, niter: int, solver: Solver | Sequence[Solver]):
        self.ax = ax
        self.niter = niter
        self.solver = solver

    def _solution(self) -> Solver:
        return self.solver if isinstance(self.solver, Solver) else max(self.solver)

    def setup(self):
        self.ax.set_title("Cost")
        self.ax.set_xlabel("iterations")
        self.ax.set_ylabel("cost")
        self.ax.set_xlim(0, self.niter)
        solution = self._solution()
        order = 10 ** np.floor(np.log10(solution.cost))
        interval = order // 2
        ylim = np.ceil((solution.cost * 1.2) / order) * order
        self.ax.set_ylim(0, ylim)
        self.ax.set_yticks(np.linspace(0, ylim, int(ylim // interval) + 1))
        (self.line,) = self.ax.plot(0, solution.cost, c="blue")

    def update(self, k: int):
        if self.line is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        xdata = np.append(self.line.get_xdata(), k)
        ydata = np.append(self.line.get_ydata(), self._solution().cost)
        self.line.set_data(xdata, ydata)
