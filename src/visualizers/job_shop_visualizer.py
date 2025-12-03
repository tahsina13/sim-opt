__all__ = ["JobShopVisualizer"]

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from optim.optimizer import Optimizer
from solvers import StochasticJobShopSolver

from .visualizer import Visualizer


class JobShopVisualizer(Visualizer):
    ax: Axes
    optim: Optimizer[StochasticJobShopSolver]
    rectangles: list[Rectangle]
    texts: list

    def __init__(
        self,
        ax: Axes,
        optim: Optimizer[StochasticJobShopSolver],
    ):
        self.ax = ax
        self.optim = optim
        self.rectangles = []
        self.texts = []

    def setup(self):
        solver = self.optim.solution
        self.ax.set_title(f"Iteration: 0, Makespan: {solver.cost:.2f}")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Machine")
        num_machines = solver.num_machines
        self.ax.set_ylim(-0.5, num_machines - 0.5)
        self.ax.set_yticks(range(num_machines))
        self.ax.set_yticklabels([f"M{i}" for i in range(num_machines)])
        self.ax.invert_yaxis() 
        self._draw_schedule()

    def _draw_schedule(self):
        for rect in self.rectangles:
            rect.remove()
        for text in self.texts:
            text.remove()
        self.rectangles.clear()
        self.texts.clear()

        solver = self.optim.solution
        if solver.solution is None or len(solver.solution) == 0:
            return

        start_times, makespan = solver._decode_schedule()
        self.ax.set_xlim(0, makespan * 1.1)
        num_jobs = len(solver.jobs)
        colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
        
        for (job_id, op_idx), start_time in start_times.items():
            op = solver.jobs[job_id][op_idx]
            rect = Rectangle(
                (start_time, op.machine_id - 0.4),
                op.processing_time,
                0.8,
                facecolor=colors[job_id],
                edgecolor='black',
                linewidth=1.5
            )
            self.ax.add_patch(rect)
            self.rectangles.append(rect)
            text = self.ax.text(
                start_time + op.processing_time / 2,
                op.machine_id,
                f"J{job_id}-{op_idx}",
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold',
                color='white' if np.mean(colors[job_id][:3]) < 0.5 else 'black'
            )
            self.texts.append(text)

    def update(self, k: int):
        solver = self.optim.solution
        self.ax.set_title(f"Iteration: {k}, Makespan: {solver.cost:.2f}")
        self._draw_schedule()