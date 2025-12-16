#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from optim import Optimizer
from optim.genetic_algorithm import GeneticAlgorithm
from optim.simulated_annealing import SimulatedAnnealing
from optim.temp_scheduler import (ExponentialTemp, LinearTemp, LogarithmicTemp,
                                  TempScheduler)
from solvers import StochasticTSPSolver, TwoOptTSPSolver
from visualizers.cost_visualizer import CostVisualizer
from visualizers.ga_visualizer import GAVisualizer
from visualizers.sa_visualizer import SAVisualizer
from visualizers.temp_visualizer import TempVisualizer
from visualizers.tsp_visualizer import TSPVisualizer

RNGS = ("generation", "selection", "combination", "mutation")

parser = argparse.ArgumentParser()
parser.add_argument("-W", "--width", help="width of world", type=int, default=16)
parser.add_argument("-H", "--height", help="height of world", type=int, default=12)
parser.add_argument("-n", "--num-nodes", help="number of nodes", type=int, default=50)
parser.add_argument("-p", "--population", help="population size", type=int, default=20)
parser.add_argument("-o", "--optimizer", help="optimizer to use", type=str)
parser.add_argument("-t", "--temperature", help="initial temperature", type=float, default=30)
parser.add_argument("-c", "--cooling-schedule", help="cooling schedule", type=str, default="exponential")
parser.add_argument("-r", "--cooling-rate", help="cooling rate", type=float, default=0.99)
parser.add_argument("-i", "--iterations", help="number of itereations", type=int, default=1500)
parser.add_argument("-s", "--seed", help="seed for simulation", type=int)

# batching
parser.add_argument("--batch", help="run in batch mode (no visualization)", action="store_true")
parser.add_argument("--output-json", help="save results to JSON file", type=str)


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
    solvers: Sequence[StochasticTSPSolver],
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

def run_batch_trial(
    num_nodes: int,
    optimizer: str,
    cooling_schedule: str,
    temperature: float,
    cooling_rate: float,
    iterations: int,
    seed: int,
    width: int = 16,
    height: int = 12,
    population: int = 20
):
    seed_seq = np.random.SeedSequence(seed)
    gen_seed, optim_seed = seed_seq.spawn(2)
    gen_rng = np.random.default_rng(gen_seed)

    xs = gen_rng.uniform(0, width, num_nodes)
    ys = gen_rng.uniform(0, height, num_nodes)
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    adj_mat = np.sqrt(dx * dx + dy * dy)
    two_opt = TwoOptTSPSolver(adj_mat)
    two_opt.solve()
    solvers = [StochasticTSPSolver(adj_mat) for _ in range(population)]
    for solver in solvers:
        solver.generate(gen_rng)

    sched = get_scheduler(cooling_schedule, temperature, cooling_rate)
    optim = get_optimizer(optimizer, solvers, sched, optim_seed)

    cost_history = []
    iterations_to_target = iterations
    converged = False
    
    for i in range(1, iterations + 1):
        optim.step()
        sched.step()
        cost_history.append(float(optim.solution.cost))
        
        if optim.solution.cost <= two_opt.cost:
            iterations_to_target = i
            converged = True
            break
    
    return {
        'problem_size': num_nodes,
        'scheduler': cooling_schedule,
        'seed': seed,
        'final_cost': float(optim.solution.cost),
        'target_cost': float(two_opt.cost),
        'iterations_to_target': int(iterations_to_target),
        'converged': bool(converged),
        'cost_history': cost_history,
    }

def main():
    args = parser.parse_args()

    # initialize RNG streams
    seed_seq = np.random.SeedSequence(args.seed)
    gen_seed, optim_seed = seed_seq.spawn(2)
    gen_rng = np.random.default_rng(gen_seed)

    # generate TSP instance and initial solution
    xs = gen_rng.uniform(0, args.width, args.num_nodes)
    ys = gen_rng.uniform(0, args.height, args.num_nodes)
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    adj_mat = np.sqrt(dx * dx + dy * dy)
    two_opt = TwoOptTSPSolver(adj_mat)
    two_opt.solve()
    solvers = [StochasticTSPSolver(adj_mat) for _ in range(args.population)]
    for solver in solvers:
        solver.generate(gen_rng)

    # initialize scheduler and optimizer
    sched = get_scheduler(args.cooling_schedule, args.temperature, args.cooling_rate)
    optim = get_optimizer(args.optimizer, solvers, sched, optim_seed)

    if args.batch:
            cost_history = []
            iterations_to_target = args.iterations 
            converged = False
            
            for i in range(1, args.iterations + 1):
                optim.step()
                sched.step()
                cost_history.append(float(optim.solution.cost))
                
                current_cost = optim.solution.cost
                if current_cost <= two_opt.cost:
                    iterations_to_target = i
                    converged = True
                    break
            
            final_cost = optim.solution.cost
            
            if args.output_json:
                results = {
                    'problem_size': args.num_nodes,
                    'scheduler': args.cooling_schedule,
                    'seed': args.seed,
                    'final_cost': float(final_cost),
                    'target_cost': float(two_opt.cost),
                    'iterations_to_target': int(iterations_to_target),
                    'converged': bool(converged),
                    'cost_history': cost_history,
                }
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
    
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        tsp_ax, cost_ax, optim_ax, temp_ax = axes.flatten()
        plt.subplots_adjust(hspace=0.3)
        fig.suptitle("Travelling Salesman Problem")

        tsp_viz = TSPVisualizer(tsp_ax, args.width, args.height, xs, ys, optim)
        tsp_viz.setup()

        cost_ax.axhline(y=two_opt.cost, c="red", ls="--", label=f"2-OPT Cost: {two_opt.cost:.2f}")
        cost_viz = CostVisualizer(cost_ax, args.iterations, optim)
        cost_viz.setup()
        cost_ax.legend()

        optim_viz = None
        if isinstance(optim, SimulatedAnnealing):
            optim_viz = SAVisualizer(optim_ax, args.iterations, optim)
        elif isinstance(optim, GeneticAlgorithm):
            optim_viz = GAVisualizer(optim_ax, args.iterations, optim)
        if optim_viz:
            optim_viz.setup()

        temp_viz = TempVisualizer(temp_ax, args.iterations, sched)
        temp_viz.setup()

        # optimization loop
        for i in range(1, args.iterations + 1):
            optim.step()
            sched.step()
            tsp_viz.update(i)
            cost_viz.update(i)
            if optim_viz:
                optim_viz.update(i)
            temp_viz.update(i)
            plt.pause(0.0001)
            time.sleep(0.001)
        plt.show()


if __name__ == "__main__":
    main()