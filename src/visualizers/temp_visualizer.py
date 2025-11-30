__all__ = ["TempVisualizer"]

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from optim.temp_scheduler import TempScheduler

from .visualizer import Visualizer


class TempVisualizer(Visualizer):
    ax: Axes
    niter: int
    temp_sched: TempScheduler
    line: Line2D | None

    def __init__(self, ax: Axes, niter: int, temp_sched: TempScheduler):
        self.ax = ax
        self.niter = niter
        self.temp_sched = temp_sched

    def setup(self):
        self.ax.set_title("Temperature")
        self.ax.set_xlabel("iterations")
        self.ax.set_ylabel("temperature")
        self.ax.set_xlim(0, self.niter)
        order = 2 ** np.floor(np.log2(self.temp_sched.temp))
        interval = order // 4
        ylim = np.ceil(self.temp_sched.temp / order) * order
        self.ax.set_ylim(0, ylim)
        self.ax.set_yticks(np.linspace(0, ylim, int(ylim // interval) + 1))
        (self.line,) = self.ax.plot(0, self.temp_sched.temp, c="blue")

    def update(self, k: int):
        if self.line is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        xdata = np.append(self.line.get_xdata(), k)
        ydata = np.append(self.line.get_ydata(), self.temp_sched.temp)
        self.line.set_xdata(xdata)
        self.line.set_ydata(ydata)
