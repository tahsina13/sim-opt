__all__ = ["KnapsackVisualizer"]

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
from optim.optimizer import Optimizer
from solvers import StochasticKnapsackSolver

from .visualizer import Visualizer


class KnapsackVisualizer(Visualizer):
    ax: Axes
    optim: Optimizer[StochasticKnapsackSolver]
    rectangles: list[Rectangle]
    texts: list

    def __init__(self, ax: Axes, optim: Optimizer[StochasticKnapsackSolver]):
        self.ax = ax
        self.optim = optim
        self.rectangles = []
        self.texts = []

    def setup(self):
        solver = self.optim.solution
        n = len(solver.weights)
        self.ax.set_title("Item Selection (Green = Selected, Red = Not Selected)")
        self.ax.set_xlabel("Item Index")
        self.ax.set_ylabel("Weight / Capacity")
        self.ax.set_xlim(-0.5, n - 0.5)
        self.ax.set_ylim(0, max(solver.weights) * 1.2)

        self._draw_bars()

    def _clear_bars(self):
        for rect in self.rectangles:
            rect.remove()
        for text in self.texts:
            text.remove()
        self.rectangles.clear()
        self.texts.clear()

    def _draw_bars(self):
        self._clear_bars()
        solver = self.optim.solution 
        sol = np.array(solver.solution, dtype=bool)
        weights = solver.weights
        for i in range(len(weights)):
            color = "green" if sol[i] else "red"
            alpha = 0.9 if sol[i] else 0.4
            rect = Rectangle(
                (i - 0.4, 0),
                0.8,              
                weights[i],      
                facecolor=color,
                edgecolor="black",
                linewidth=1.0,
                alpha=alpha
            )
            self.ax.add_patch(rect)
            self.rectangles.append(rect)
        current_weight = np.sum(weights[sol])
        current_value = solver.cost
        usage_pct = 100 * current_weight / solver.limit
        summary = self.ax.text(
            len(weights) * 0.5, max(weights) * 1.1,
            f'Weight: {current_weight}/{solver.limit} ({usage_pct:.1f}%) | Value: {current_value:.0f}',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        self.texts.append(summary)

    def update(self, k: int):
        self._draw_bars()