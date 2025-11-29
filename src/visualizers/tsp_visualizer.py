__all__ = ["TSPVisualizer"]

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from solvers import TSPSolver

from .visualizer import Visualizer


class TSPVisualizer(Visualizer):
    ax: Axes
    width: int
    height: int
    xs: npt.NDArray[np.float64]
    ys: npt.NDArray[np.float64]
    solver: TSPSolver
    line: Line2D | None

    def __init__(
        self,
        ax: Axes,
        width: int,
        height: int,
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        solver: TSPSolver,
    ):
        self.ax = ax
        self.width = width
        self.height = height
        self.xs = xs
        self.ys = ys
        self.solver = solver

    def _indices(self) -> npt.NDArray[np.int_]:
        if self.solver.solution is None or len(self.solver.solution) == 0:
            return np.array([], dtype=np.int_)
        return np.append(self.solver.solution, self.solver.solution[0])

    def setup(self):
        self.ax.set_title(f"Iteration: 0, Cost: {self.solver.cost:.2f}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(-0.5, self.width + 0.5)
        self.ax.set_ylim(-0.5, self.height + 0.5)
        self.ax.tick_params(top=True, bottom=True, left=True, right=True)
        self.ax.scatter(self.xs, self.ys, c="black")
        indices = self._indices()
        (self.line,) = self.ax.plot(self.xs[indices], self.ys[indices], c="black")

    def update(self, k: int):
        if self.line is None:
            raise RuntimeError(
                "Failed to update visualizer because visualizer was not setup"
            )
        self.ax.set_title(f"Iteration: {k}, Cost: {self.solver.cost:.2f}")
        indices = self._indices()
        self.line.set_data(self.xs[indices], self.ys[indices])
