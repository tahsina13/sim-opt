#!/usr/bin/env python3

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from optim import Optimizer
from optim.simulated_annealing import SimulatedAnnealing
from optim.temp_scheduler import (ExponentialTemp, LinearTemp, LogarithmicTemp,
                                  TempScheduler)
from solvers import StochasticTSPSolver, TwoOptTSPSolver
from visualizers.tsp_visualizer import TSPVisualizer

RNGS = ("generation", "selection", "combination", "mutation")

parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width", help="width of world", type=int, default=16)
parser.add_argument("-H", "--height", help="height of world", type=int, default=12)
parser.add_argument("-n", "--num-nodes", help="number of nodes", type=int, default=50)
parser.add_argument("-o", "--optimizer", help="optimizer to use", type=str)
parser.add_argument("-t", "--temperature", help="initial temperature", type=float, default=30)
parser.add_argument("-c", "--cooling-schedule", help="cooling schedule", type=str, default="exponential")
parser.add_argument("-r", "--cooling-rate", help="cooling rate", type=float, default=0.99)
parser.add_argument("-i", "--iterations", help="number of itereations", type=int, default=1500)
parser.add_argument("-s", "--seed", help="seed for simulation", type=int)


def get_scheduler(
    cooling_schedule: str, temp: float, cooling_rate: float
) -> TempScheduler:
    match cooling_schedule:
        case "linear":
            return LinearTemp(temp, cooling_rate)
        case "exponential":
            return ExponentialTemp(temp, cooling_rate)
        case "logarithmic":
            return LogarithmicTemp(temp)
        case _:
            raise ValueError(f"Unkown cooling schedule '{cooling_schedule}'")


def get_optimizer(
    optimizer: str,
    solver: StochasticTSPSolver,
    temp_sched: TempScheduler,
    rngs: dict[str, np.random.Generator],
) -> Optimizer:
    match optimizer:
        case "sa":
            return SimulatedAnnealing(solver, temp_sched, rngs)
        case "ga":
            raise NotImplementedError(f"Optimizer '{optimizer}' not implemented yet")
        case _:
            raise ValueError(f"Unknown optimizer '{optimizer}'")


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
    two_opt = TwoOptTSPSolver(adj_mat)
    two_opt.solve()

    # initialize scheduler and optimizer
    sched = get_scheduler(args.cooling_schedule, args.temperature, args.cooling_rate)
    optim = get_optimizer(args.optimizer, solver, sched, rngs)

    # initialize visualizers
    fig, ax = plt.subplots(1, 1,)
    fig.suptitle(f"2-OPT Cost: {two_opt.cost:.2f}", c="red")
    if two_opt.solution is None or len(two_opt.solution) == 0:
        two_opt_indices = np.array([], dtype=np.int_)
    else:
        two_opt_indices = np.append(two_opt.solution, two_opt.solution[0])
    ax.plot(xs[two_opt_indices], ys[two_opt_indices], c="red", ls="--")
    visualizer = TSPVisualizer(ax, args.width, args.height, xs, ys, solver)
    visualizer.setup()

    # optimization loop
    for i in range(1, args.iterations + 1):
        optim.step()
        sched.step()
        visualizer.update(i)
        plt.pause(0.0001)
        time.sleep(0.1)
    plt.show()


if __name__ == "__main__":
    main()
