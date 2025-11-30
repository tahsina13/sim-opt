__all__ = ["GAVisualizer"]

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from optim.genetic_algorithm import GeneticAlgorithm

from .visualizer import Visualizer


class GAVisualizer(Visualizer):
    ax: Axes
    niter: int
    optim: GeneticAlgorithm
    scatter: PathCollection | None

    def __init__(self, ax: Axes, niter: int, optim: GeneticAlgorithm):
        self.ax = ax
        self.niter = niter
        self.optim = optim

    def _gini(self):
        return 1 - np.sum(np.square(self.optim.probs))

    def setup(self):
        self.ax.set_title("Gini Index")
        self.ax.set_xlabel("iterations")
        self.ax.set_ylabel("gini index")
        self.ax.set_xlim(0, self.niter)
        self.ax.set_ylim(0, 1.1)
        self.scatter = self.ax.scatter(0, self._gini(), c="blue", s=10)

    def update(self, k: int):
        if self.scatter is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        points = np.vstack([self.scatter.get_offsets(), [k, self._gini()]])
        self.scatter.set_offsets(points)
