__all__ = ["SAVisualizer"]

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from optim.simulated_annealing import SimulatedAnnealing

from .visualizer import Visualizer


class SAVisualizer(Visualizer):
    ax: Axes
    niter: int
    optim: SimulatedAnnealing
    scatter: PathCollection | None

    def __init__(self, ax: Axes, niter: int, optim: SimulatedAnnealing):
        self.ax = ax
        self.niter = niter
        self.optim = optim

    def setup(self):
        self.ax.set_title("Acceptance Probability")
        self.ax.set_xlabel("iterations")
        self.ax.set_ylabel("probability")
        self.ax.set_xlim(0, self.niter)
        self.ax.set_ylim(0, 1.1)
        self.scatter = self.ax.scatter(0, self.optim.prob, c="blue", s=10)

    def update(self, k: int):
        if self.scatter is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        points = np.vstack([self.scatter.get_offsets(), [k, self.optim.prob]])
        self.scatter.set_offsets(points)
