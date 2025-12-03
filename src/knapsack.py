import argparse
import time
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from optim import Optimizer
from optim.genetic_algorithm import GeneticAlgorithm
from optim.simulated_annealing import SimulatedAnnealing
from optim.temp_scheduler import (ExponentialTemp, LinearTemp, LogarithmicTemp,
                                  TempScheduler)
from solvers import StochasticKnapsackSolver
from visualizers.cost_visualizer import CostVisualizer
from visualizers.ga_visualizer import GAVisualizer
from visualizers.sa_visualizer import SAVisualizer
from visualizers.temp_visualizer import TempVisualizer

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-items", help="number of items", type=int, default=50)
parser.add_argument("--min-weight", help="minimum item weight", type=int, default=1)
parser.add_argument("--max-weight", help="maximum item weight", type=int, default=50)
parser.add_argument("--capacity-ratio", help="knapsack capacity as ratio of total weight", type=float, default=0.5)
parser.add_argument("-p", "--population", help="population size", type=int, default=20)
parser.add_argument("-o", "--optimizer", help="optimizer to use", type=str, required=True)
parser.add_argument("-t", "--temperature", help="initial temperature", type=float, default=50)
parser.add_argument("-c", "--cooling-schedule", help="cooling schedule", type=str, default="exponential")
parser.add_argument("-r", "--cooling-rate", help="cooling rate", type=float, default=0.99)
parser.add_argument("-i", "--iterations", help="number of iterations", type=int, default=1500)
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
            raise ValueError(f"Unknown cooling schedule '{cooling_schedule}'")


def get_optimizer(
    optimizer: str,
    solvers: Sequence[StochasticKnapsackSolver],
    temp_sched: TempScheduler,
    seed_seq: np.random.SeedSequence,
) -> Optimizer:
    match optimizer:
        case "sa":
            return SimulatedAnnealing(max(solvers), temp_sched, seed_seq)
        case "ga":
            return GeneticAlgorithm(solvers, temp_sched, seed_seq)
        case _:
            raise ValueError(f"Unknown optimizer '{optimizer}'")


def greedy_knapsack(weights: np.ndarray, values: np.ndarray, limit: int) -> float:
    ratios = values / weights
    sorted_indices = np.argsort(ratios)[::-1]
    total_weight = 0
    total_value = 0
    for idx in sorted_indices:
        if total_weight + weights[idx] <= limit:
            total_weight += weights[idx]
            total_value += values[idx]
    return float(total_value)

def main():
    args = parser.parse_args()

    # initialize RNG streams
    seed_seq = np.random.SeedSequence(args.seed)
    gen_seed, optim_seed = seed_seq.spawn(2)
    gen_rng = np.random.default_rng(gen_seed)

    # generate knapsack instance
    weights = gen_rng.integers(args.min_weight, args.max_weight + 1, args.num_items)
    values = gen_rng.integers(args.min_weight, args.max_weight + 1, args.num_items) 
    total_weight = np.sum(weights)
    limit = int(total_weight * args.capacity_ratio)
    
    print(f"Generated {args.num_items} items")
    print(f"Total weight: {total_weight}, Capacity: {limit} ({args.capacity_ratio*100:.0f}%)")
    
    # compute greedy solution
    greedy_cost = greedy_knapsack(weights, values, limit)  # ← Add 'values' here
    print(f"Greedy solution: {greedy_cost:.2f}")
    
    # generate initial population
    solvers = [StochasticKnapsackSolver(weights, values, limit) for _ in range(args.population)]  # ← Add 'values' here
    for solver in solvers:
        solver.generate(gen_rng)

    # initialize scheduler and optimizer
    sched = get_scheduler(args.cooling_schedule, args.temperature, args.cooling_rate)
    optim = get_optimizer(args.optimizer, solvers, sched, optim_seed)

    # initialize visualizers
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.3)
    fig.suptitle("0/1 Knapsack Problem")

    # cost visualization
    cost_ax = axes[0]
    cost_ax.axhline(y=greedy_cost, c="red", ls="--", label=f"Greedy Cost: {greedy_cost:.2f}")
    cost_viz = CostVisualizer(cost_ax, args.iterations, optim)
    cost_viz.setup()
    cost_ax.legend()

    # optimizer-specific visualization
    optim_ax = axes[1]
    optim_viz = None
    if isinstance(optim, SimulatedAnnealing):
        optim_viz = SAVisualizer(optim_ax, args.iterations, optim)
    elif isinstance(optim, GeneticAlgorithm):
        optim_viz = GAVisualizer(optim_ax, args.iterations, optim)
    if optim_viz:
        optim_viz.setup()

    # temperature visualization
    temp_ax = axes[2]
    temp_viz = TempVisualizer(temp_ax, args.iterations, sched)
    temp_viz.setup()

    # capacity utilization
    util_ax = axes[3]
    util_ax.set_title("Capacity Utilization (Weight)")
    util_ax.set_xlabel("iterations")
    util_ax.set_ylabel("utilization %")
    util_ax.set_xlim(0, args.iterations)
    util_ax.set_ylim(0, 110)
    util_ax.axhline(y=100, c="red", ls="--", label="Capacity")
    
    # initial weight utilization
    initial_weight = np.sum(optim.solution.weights[np.flatnonzero(optim.solution.solution)])
    util_line, = util_ax.plot(0, 100 * initial_weight / limit, c="blue")
    util_ax.legend()

    # optimization loop
    for i in range(1, args.iterations + 1):
        optim.step()
        sched.step()
        cost_viz.update(i)
        if optim_viz:
            optim_viz.update(i)
        temp_viz.update(i)
        
        # Update utilization (weight, not value!)
        current_weight = np.sum(optim.solution.weights[np.flatnonzero(optim.solution.solution)])
        xdata = np.append(util_line.get_xdata(), i)
        ydata = np.append(util_line.get_ydata(), 100 * current_weight / limit)
        util_line.set_data(xdata, ydata)
        
        plt.pause(0.0001)
        time.sleep(0.001)
    
    plt.show()


if __name__ == "__main__":
    main()