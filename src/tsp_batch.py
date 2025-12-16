#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from tsp import run_batch_trial

PROBLEM_SIZES = [10, 20, 30, 50, 75, 100, 150]
NUM_INSTANCES = 30
SCHEDULERS = ["linear", "exponential", "logarithmic"]
INITIAL_TEMP = 32.0
COOLING_RATE = 0.997
MAX_ITERS = 2000
BASE_SEED = 42

def run_single_trial(size, scheduler, instance_id):
    seed = BASE_SEED + size * 1000 + instance_id
    output_file = f"results/raw/n{size}_{scheduler}_{instance_id}.json"
    
    results = run_batch_trial(
        num_nodes=size,
        optimizer="sa",
        cooling_schedule=scheduler,
        temperature=INITIAL_TEMP,
        cooling_rate=COOLING_RATE,
        iterations=MAX_ITERS,
        seed=seed
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def create_fixed_iteration_plot():
    CHECK_ITERATIONS = [100, 500, 1000, 1500, 2000]
    
    fig, axes = plt.subplots(1, len(CHECK_ITERATIONS), figsize=(20, 4))
    
    for iter_idx, target_iter in enumerate(CHECK_ITERATIONS):
        ax = axes[iter_idx]
        
        for scheduler in SCHEDULERS:
            costs_by_size = []
            
            for size in PROBLEM_SIZES:
                costs_at_iter = []
                for instance_id in range(NUM_INSTANCES):
                    with open(f"results/raw/n{size}_{scheduler}_{instance_id}.json", 'r') as f:
                        data = json.load(f)
                        history = data['cost_history']
                        if target_iter <= len(history):
                            costs_at_iter.append(history[target_iter - 1])
                        else:
                            costs_at_iter.append(history[-1])
                
                avg_cost = sum(costs_at_iter) / len(costs_at_iter)
                costs_by_size.append(avg_cost)
            
            ax.plot(PROBLEM_SIZES, costs_by_size, marker='o', label=scheduler, linewidth=2, markersize=8)
        
        ax.set_xlabel('Problem Size (n)')
        ax.set_ylabel('Average Cost')
        ax.set_title(f'After {target_iter} Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PROBLEM_SIZES)
    
    plt.tight_layout()
    plt.savefig('results/fixed_iteration_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary():
    results = defaultdict(lambda: defaultdict(list))
    
    for size in PROBLEM_SIZES:
        for scheduler in SCHEDULERS:
            for instance_id in range(NUM_INSTANCES):
                with open(f"results/raw/n{size}_{scheduler}_{instance_id}.json", 'r') as f:
                    data = json.load(f)
                    results[size][scheduler].append(data['iterations_to_target'])
    
    summary = {}
    for size in PROBLEM_SIZES:
        summary[size] = {}
        for scheduler in SCHEDULERS:
            iters = results[size][scheduler]
            summary[size][scheduler] = sum(iters) / len(iters)
    
    with open('results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    plt.figure(figsize=(10, 6))
    for scheduler in SCHEDULERS:
        avg_iters = [summary[size][scheduler] for size in PROBLEM_SIZES]
        plt.plot(PROBLEM_SIZES, avg_iters, marker='o', label=scheduler, linewidth=2, markersize=8)
    
    plt.xlabel('Problem Size (n)')
    plt.ylabel('Average Iterations to Target')
    plt.title('Cooling Schedule Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(PROBLEM_SIZES)
    plt.savefig('results/comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    
    for size in PROBLEM_SIZES:
        for instance_id in range(NUM_INSTANCES):
            for scheduler in SCHEDULERS:
                run_single_trial(size, scheduler, instance_id)
    
    create_summary()
    create_fixed_iteration_plot()

if __name__ == "__main__":
    main()