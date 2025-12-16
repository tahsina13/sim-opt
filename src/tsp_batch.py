#!/usr/bin/env python3

import subprocess
import json
from pathlib import Path
from collections import defaultdict

PROBLEM_SIZES = [10, 20, 30]
NUM_INSTANCES = 10 
SCHEDULERS = ["linear", "exponential", "logarithmic"]
INITIAL_TEMP = 40.0
COOLING_RATE = 0.995
MAX_ITERS = 2000
BASE_SEED = 42

def run_single_trial(size, scheduler, instance_id):
    seed = BASE_SEED + size * 1000 + instance_id
    output_file = f"results/raw/n{size}_{scheduler}_{instance_id}.json"
    cmd = [
        "python", "tsp.py",
        "-n", str(size),
        "-o", "sa",
        "-c", scheduler,
        "-t", str(INITIAL_TEMP),
        "-r", str(COOLING_RATE),
        "-i", str(MAX_ITERS),
        "-s", str(seed),
        "--batch",
        "--output-json", output_file
    ]
    subprocess.run(cmd, capture_output=True, text=True)

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

def main():
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    
    for size in PROBLEM_SIZES:
        for instance_id in range(NUM_INSTANCES):
            for scheduler in SCHEDULERS:
                run_single_trial(size, scheduler, instance_id)
    
    create_summary()

if __name__ == "__main__":
    main()