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
from solvers.job_shop_solver import (GreedyJobShopSolver, Operation,
                                     StochasticJobShopSolver)
from visualizers.cost_visualizer import CostVisualizer
from visualizers.ga_visualizer import GAVisualizer
from visualizers.job_shop_visualizer import JobShopVisualizer
from visualizers.sa_visualizer import SAVisualizer
from visualizers.temp_visualizer import TempVisualizer

RNGS = ("generation", "selection", "combination", "mutation")

parser = argparse.ArgumentParser()
parser.add_argument("-j", "--num-jobs", help="number of jobs", type=int, default=5)
parser.add_argument("-m", "--num-machines", help="number of machines", type=int, default=3)
parser.add_argument("--min-ops", help="minimum operations per job", type=int, default=2)
parser.add_argument("--max-ops", help="maximum operations per job", type=int, default=5)
parser.add_argument("--min-time", help="minimum processing time", type=float, default=1.0)
parser.add_argument("--max-time", help="maximum processing time", type=float, default=10.0)
parser.add_argument("-p", "--population", help="population size", type=int, default=20)
parser.add_argument("-o", "--optimizer", help="optimizer to use", type=str, required=True)
parser.add_argument("-t", "--temperature", help="initial temperature", type=float, default=100)
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
    solvers: Sequence[StochasticJobShopSolver],
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


def generate_job_shop_instance(
    num_jobs: int,
    num_machines: int,
    min_ops: int,
    max_ops: int,
    min_time: float,
    max_time: float,
    rng: np.random.Generator
) -> list[list[Operation]]:
    jobs = []
    
    for job_id in range(num_jobs):
        num_ops = rng.integers(min_ops, max_ops + 1)
        machines = rng.choice(num_machines, size=num_ops, replace=True)
        
        operations = []
        for _, machine_id in enumerate(machines):
            processing_time = rng.uniform(min_time, max_time)
            operations.append(Operation(job_id, machine_id, processing_time))
        
        jobs.append(operations)
    
    return jobs


def main():
    args = parser.parse_args()

    # initialize RNG streams
    seed_seq = np.random.SeedSequence(args.seed)
    gen_seed, optim_seed = seed_seq.spawn(2)
    gen_rng = np.random.default_rng(gen_seed)

    # generate job shop instance
    jobs = generate_job_shop_instance(
        args.num_jobs,
        args.num_machines,
        args.min_ops,
        args.max_ops,
        args.min_time,
        args.max_time,
        gen_rng
    )
    
    # compute greedy solution
    greedy = GreedyJobShopSolver(jobs, args.num_machines)
    greedy.solve()
    print(f"\nCritical Path heuristic makespan: {greedy.cost:.2f}")
    
    # generate initial population
    solvers = [StochasticJobShopSolver(jobs, args.num_machines) for _ in range(args.population)]
    for solver in solvers:
        solver.generate(gen_rng)

    # initialize scheduler and optimizer
    sched = get_scheduler(args.cooling_schedule, args.temperature, args.cooling_rate)
    optim = get_optimizer(args.optimizer, solvers, sched, optim_seed)

    print(f"Initial best solution makespan: {optim.solution.cost:.2f}\n")

    # initialize visualizers
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    job_ax, cost_ax, optim_ax, temp_ax = axes.flatten()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle("Job Shop Scheduling Problem")

    job_viz = JobShopVisualizer(job_ax, optim)
    job_viz.setup()

    cost_ax.axhline(y=greedy.cost, c="red", ls="--", label=f"Critical Path: {greedy.cost:.2f}")
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
        job_viz.update(i)
        cost_viz.update(i)
        if optim_viz:
            optim_viz.update(i)
        temp_viz.update(i)
        plt.pause(0.0001)
        time.sleep(0.001)

if __name__ == "__main__":
    main()