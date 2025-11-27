#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from optim import Optimizer
from optim.simulated_annealing import SimulatedAnnealing
from optim.temp_scheduler import (ExponentialTemp, LinearTemp, LogarithmicTemp,
                                  TempScheduler)
from solvers import StochasticTSPSolver

RNGS = ("generation", "selection", "combination", "mutation")

parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width", help="width of world", type=int, default=16)
parser.add_argument("-H", "--height", help="height of world", type=int, default=12)
parser.add_argument("-n", "--num-nodes", help="number of nodes", type=int, default=120)
parser.add_argument("-o", "--optimizer", help="optimizer to use", type=str)
parser.add_argument("-t", "--temperature", help="initial temperature", type=float, default=50)
parser.add_argument("-c", "--cooling-schedule", help="cooling schedule", type=str, default="linear")
parser.add_argument("-r", "--cooling-rate", help="cooling rate", type=float, default=1.0)
parser.add_argument("-i", "--iterations", help="number of itereations", type=int, default=50)
parser.add_argument("-s", "--seed", help="seed for simulation", type=int)


def get_optimizer(
    optimizer: str, solver: StochasticTSPSolver, temp: float, rngs: dict[str, np.random.Generator]
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
    sol_xs = xs[solver.solution] if solver.solution else []
    sol_ys = ys[solver.solution] if solver.solution else []
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
    plt.ion()
    visualize_tsp(args.width, args.height, xs, ys, solver)
    for i in range(args.iterations):
        optim.step()
        sched.step()
        plt.title(f"Iteration: {i + 1}")
        visualize_tsp(args.width, args.height, xs, ys, solver)
        plt.pause(0.5)
    plt.show()


if __name__ == "__main__":
    main()
