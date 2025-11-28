#!/usr/bin/env python3

import argparse
import time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optim import Optimizer
from optim.simulated_annealing import SimulatedAnnealing
from optim.temp_scheduler import (ExponentialTemp, LinearTemp, LogarithmicTemp,
                                  TempScheduler)
from solvers import StochasticTSPSolver

RNGS = ("generation", "selection", "combination", "mutation")

config = [
    ("-W", "--width", {"type": int, "default": 16, "help": "width of world"}),
    ("-H", "--height", {"type": int, "default": 12, "help": "height of world"}),
    ("-n", "--num-nodes", {"type": int, "default": 120, "help": "number of nodes"}),
    ("-o", "--optimizer", {"type": str, "help": "optimizer to use"}),
    ("-t", "--temperature", {"type": float, "default": 50, "help": "initial temperature"}),
    ("-c", "--cooling-schedule", {"type": str, "default": "linear", "help": "cooling schedule"}),
    ("-r", "--cooling-rate", {"type": float, "default": 1.0, "help": "cooling rate"}),
    ("-i", "--iterations", {"type": int, "default": 50, "help": "number of iterations"}),
    ("-s", "--seed", {"type": int, "help": "seed for simulation"}),
]

parser = argparse.ArgumentParser()
for short, long, kw in config:
    parser.add_argument(short, long, **kw)


def get_optimizer(
    optimizer: str,
    solver: StochasticTSPSolver,
    temp: float,
    rngs: dict[str, np.random.Generator],
) -> Optimizer:
    match optimizer:
        case "sa":
            return SimulatedAnnealing(solver, temp, rngs)
        case "ga":
            raise NotImplementedError(f"Optimizer '{optimizer}' not implemented yet")
        case _:
            raise ValueError(f"Unknown optimizer '{optimizer}'")


def get_scheduler(
    cooling_schedule: str, optimizer: Optimizer, cooling_rate: float
) -> TempScheduler:
    match cooling_schedule:
        case "linear":
            return LinearTemp(optimizer, cooling_rate)
        case "exponential":
            return ExponentialTemp(optimizer, cooling_rate)
        case "logarithmic":
            return LogarithmicTemp(optimizer)
        case _:
            raise ValueError(f"Unkown cooling schedule '{cooling_schedule}'")


def visualize_tsp(
    width: int,
    height: int,
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
    solver: StochasticTSPSolver,
):
    indices = np.append(solver.solution, solver.solution[0]) if solver.solution else []
    sol_xs, sol_ys = xs[indices], ys[indices]
    plt.xlim(-0.5, width + 0.5)
    plt.ylim(-0.5, height + 0.5)
    plt.tick_params(top=True, bottom=True, left=True, right=True)
    plt.scatter(xs, ys, c="black")
    plt.plot(sol_xs, sol_ys, c="black")


def main():
    args = parser.parse_args()

    # initialize RNG streams
    sq = np.random.SeedSequence(args.seed)
    rngs = {
        name: np.random.default_rng(sd) for name, sd in zip(RNGS, sq.spawn(len(RNGS)))
    }

    # generate TSP instance and initial solution
    xs = rngs["generation"].uniform(0, args.width, args.num_nodes)
    ys = rngs["generation"].uniform(0, args.height, args.num_nodes)
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    adj_mat = np.sqrt(dx * dx + dy * dy)
    solver = StochasticTSPSolver(adj_mat)
    solver.generate(rngs["generation"])

    # initialize optimizer
    optim = get_optimizer(args.optimizer, solver, args.temperature, rngs)
    sched = get_scheduler(args.cooling_schedule, optim, args.cooling_rate)

    # optimization loop
    solution = cast(StochasticTSPSolver, optim.solution)
    visualize_tsp(args.width, args.height, xs, ys, solution)
    for i in range(1, args.iterations + 1):
        optim.step()
        sched.step()
        solution = cast(StochasticTSPSolver, optim.solution)
        plt.clf()
        plt.title(f"Iteration: {i}, Cost: {solution.cost:.2f}")
        solution = cast(StochasticTSPSolver, solution)
        visualize_tsp(args.width, args.height, xs, ys, solution)
        plt.pause(0.0001)
        time.sleep(0.1)
    plt.show()


if __name__ == "__main__":
    main()
