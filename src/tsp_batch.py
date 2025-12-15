#!/usr/bin/env python3

import subprocess
import json
from pathlib import Path

# Configuration
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
    return output_file

def main():
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    total = len(PROBLEM_SIZES) * len(SCHEDULERS) * NUM_INSTANCES
    count = 0
    
    for size in PROBLEM_SIZES:
        for instance_id in range(NUM_INSTANCES):
            for scheduler in SCHEDULERS:
                count += 1
                print(f"Test {count}/{total}")
                run_single_trial(size, scheduler, instance_id)

if __name__ == "__main__":
    main()