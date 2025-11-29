__all__ = ["CostVisualizer"]

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from solvers import Solver

from .visualizer import Visualizer


class CostVisualizer(Visualizer):
    ax: Axes
    niter: int
    solver: Solver
    line: Line2D | None

    def __init__(self, ax: Axes, niter: int, solver: Solver):
        self.ax = ax
        self.niter = niter
        self.solver = solver

    def setup(self):
        self.ax.set_title("Cost")
        self.ax.set_xlabel("iterations")
        self.ax.set_ylabel("cost")
        self.ax.set_xlim(0, self.niter)
        order = 10 ** np.floor(np.log10(self.solver.cost))
        interval = order // 2
        ylim = np.ceil((self.solver.cost * 1.2) / order) * order
        self.ax.set_ylim(0, ylim)
        self.ax.set_yticks(np.linspace(0, ylim, int(ylim // interval) + 1))
        (self.line,) = self.ax.plot(0, self.solver.cost, c="blue")

    def update(self, k: int):
        if self.line is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        xdata = np.append(self.line.get_xdata(), k)
        ydata = np.append(self.line.get_ydata(), self.solver.cost)
        self.line.set_data(xdata, ydata)
