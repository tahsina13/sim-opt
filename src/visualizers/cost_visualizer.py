__all__ = ["CostVisualizer"]

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from optim.optimizer import Optimizer

from .visualizer import Visualizer


class CostVisualizer(Visualizer):
    ax: Axes
    niter: int
    optim: Optimizer
    line: Line2D | None

    def __init__(self, ax: Axes, niter: int, optim: Optimizer):
        self.ax = ax
        self.niter = niter
        self.optim = optim
        self.line = None

    def setup(self):
        self.ax.set_title("Cost")
        self.ax.set_xlabel("iterations")
        self.ax.set_ylabel("cost")
        self.ax.set_xlim(0, self.niter)
        initial_cost = self.optim.solution.cost
        order = 10 ** np.floor(np.log10(max(abs(initial_cost), 1)))
        ylim = np.ceil((initial_cost * 1.2) / order) * order
        self.ax.set_ylim(0, ylim)
        (self.line,) = self.ax.plot(0, initial_cost, c="blue")

    def update(self, k: int):
        if self.line is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        xdata = np.append(self.line.get_xdata(), k)
        ydata = np.append(self.line.get_ydata(), self.optim.solution.cost)
        self.line.set_data(xdata, ydata)
        current_cost = self.optim.solution.cost
        _, current_ylim = self.ax.get_ylim()
        if current_cost > current_ylim * 0.9:
            order = 10 ** np.floor(np.log10(max(abs(current_cost), 1)))
            new_ylim = np.ceil((current_cost * 1.2) / order) * order
            self.ax.set_ylim(0, new_ylim)
            interval = order // 2
            self.ax.set_yticks(np.linspace(0, new_ylim, int(new_ylim // interval) + 1))