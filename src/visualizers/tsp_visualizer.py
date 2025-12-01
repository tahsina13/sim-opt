__all__ = ["TSPVisualizer"]

from typing import Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from optim.optimizer import Optimizer
from solvers import StochasticTSPSolver

from .visualizer import Visualizer


class TSPVisualizer(Visualizer):
    ax: Axes
    width: int
    height: int
    xs: npt.NDArray[np.float64]
    ys: npt.NDArray[np.float64]
    optim: Optimizer[StochasticTSPSolver]
    line: Line2D | None

    def __init__(
        self,
        ax: Axes,
        width: int,
        height: int,
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        optim: Optimizer[StochasticTSPSolver],
    ):
        self.ax = ax
        self.width = width
        self.height = height
        self.xs = xs
        self.ys = ys
        self.optim = optim

    def _points(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        solver = self.optim.solution
        if solver.solution is None or len(solver.solution) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        indices = np.append(solver.solution, solver.solution[0])
        return self.xs[indices], self.ys[indices]

    def setup(self):
        xs, ys = self._points()
        self.ax.set_title(f"Iteration: 0, Cost: {self.optim.solution.cost:.2f}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(-0.5, self.width + 0.5)
        self.ax.set_ylim(-0.5, self.height + 0.5)
        self.ax.tick_params(top=True, bottom=True, left=True, right=True)
        self.ax.scatter(self.xs, self.ys, c="black")
        (self.line,) = self.ax.plot(xs, ys, c="black")

    def update(self, k: int):
        if self.line is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        xs, ys = self._points()
        self.ax.set_title(f"Iteration: {k}, Cost: {self.optim.solution.cost:.2f}")
        self.line.set_data(xs, ys)
