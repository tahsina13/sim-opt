__all__ = ["JobShopSolver", "StochasticJobShopSolver", "GreedyJobShopSolver"]

import sys
from typing import cast

import numpy as np

from .solver import HeuristicSolver, Solver, StochasticSolver


class Operation:
    job_id: int
    machine_id: int
    processing_time: float
    
    def __init__(self, job_id: int, machine_id: int, processing_time: float):
        self.job_id = job_id
        self.machine_id = machine_id
        self.processing_time = processing_time


class JobShopSolver(Solver):
    jobs: list[list[Operation]]
    num_machines: int
    solution: list[tuple[int, int]] | None
    
    def __init__(self, jobs: list[list[Operation]], num_machines: int):
        self.jobs = jobs
        self.num_machines = num_machines
        self.solution = None
    
    def __eq__(self, other: object) -> bool:
        other = cast(JobShopSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to equate solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost == other.cost

    def __gt__(self, other: Solver) -> bool:
        other = cast(JobShopSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to compare solution '{self.solution}' with '{other.solution}'"
            )
        return self.cost < other.cost
    
    def _decode_schedule(self) -> tuple[dict[tuple[int, int], float], float]:
        if self.solution is None:
            return {}, sys.float_info.max
        
        machine_available = [0.0] * self.num_machines
        job_available = [0.0] * len(self.jobs)
        job_next_op = [0] * len(self.jobs)
        start_times = {}
        
        for job_id, op_idx in self.solution:
            if (job_id, op_idx) in start_times:
                continue              
            if op_idx != job_next_op[job_id]:
                continue  
            op = self.jobs[job_id][op_idx]
            
            # op starts after previous operation in job and machine are done
            prev_op_done = job_available[job_id]
            machine_ready = machine_available[op.machine_id]      
            start_time = max(prev_op_done, machine_ready)
            end_time = start_time + op.processing_time
            start_times[(job_id, op_idx)] = start_time
            machine_available[op.machine_id] = end_time
            job_available[job_id] = end_time
            job_next_op[job_id] += 1
        
        # schedule any remaining operations
        for job_id, job_ops in enumerate(self.jobs):
            for op_idx in range(job_next_op[job_id], len(job_ops)):
                op = job_ops[op_idx]
                prev_op_done = job_available[job_id]
                machine_ready = machine_available[op.machine_id]
                start_time = max(prev_op_done, machine_ready)
                end_time = start_time + op.processing_time
                start_times[(job_id, op_idx)] = start_time
                machine_available[op.machine_id] = end_time
                job_available[job_id] = end_time
        
        makespan = max(machine_available) if machine_available else 0.0
        return start_times, makespan
    
    @property
    def cost(self) -> float:
        _, makespan = self._decode_schedule()
        return makespan


class GreedyJobShopSolver(JobShopSolver, HeuristicSolver):
    def solve(self):
        # Precompute tail times for each operation
        tail_times = {}
        for job_id, job_ops in enumerate(self.jobs):
            cumulative = 0.0
            for op_idx in range(len(job_ops) - 1, -1, -1):
                op = job_ops[op_idx]
                tail_times[(job_id, op_idx)] = cumulative
                cumulative += op.processing_time

        # Track next operation per job and machine availability
        job_next_op = [0] * len(self.jobs)
        machine_available = [0.0] * self.num_machines
        job_available = [0.0] * len(self.jobs)
        solution = []

        total_ops = sum(len(job) for job in self.jobs)

        while len(solution) < total_ops:
            available_ops = []

            # Collect all operations whose previous job operations are done
            for job_id, job_ops in enumerate(self.jobs):
                op_idx = job_next_op[job_id]
                if op_idx < len(job_ops):
                    op = job_ops[op_idx]
                    # Earliest start time considering machine & job availability
                    est = max(job_available[job_id], machine_available[op.machine_id])
                    priority = tail_times[(job_id, op_idx)] + op.processing_time
                    available_ops.append((priority, est, job_id, op_idx, op))

            if not available_ops:
                break

            # Choose operation with largest priority (tie-breaker: earliest start)
            available_ops.sort(key=lambda x: (-x[0], x[1]))
            _, est, job_id, op_idx, op = available_ops[0]

            # Schedule operation
            solution.append((job_id, op_idx))
            start_time = max(job_available[job_id], machine_available[op.machine_id])
            end_time = start_time + op.processing_time
            job_available[job_id] = end_time
            machine_available[op.machine_id] = end_time
            job_next_op[job_id] += 1

        self.solution = solution


class StochasticJobShopSolver(JobShopSolver, StochasticSolver):
    def generate(self, rng: np.random.Generator):
        all_ops = []
        for job_id, job_ops in enumerate(self.jobs):
            for op_idx in range(len(job_ops)):
                all_ops.append((job_id, op_idx))

        # shuffle to make initial solution worse
        job_blocks = {}
        for op in all_ops:
            job_blocks.setdefault(op[0], []).append(op)

        shuffled_ops = []
        job_ids = list(job_blocks.keys())
        rng.shuffle(job_ids)
        for job_id in job_ids:
            ops = job_blocks[job_id]
            rng.shuffle(ops)
            shuffled_ops.extend(ops)

        self.solution = shuffled_ops

    def mutate(self, rng: np.random.Generator):
        if self.solution is None:
            raise RuntimeError(f"Failed to mutate solution '{self.solution}'")
        i, j = sorted(rng.choice(len(self.solution), size=2, replace=False))
        segment = self.solution[i:j]
        job_groups = {}
        for op in segment:
            job_groups.setdefault(op[0], []).append(op)
        
        reversed_segment = []
        for job_id in reversed(list(job_groups.keys())):
            reversed_segment.extend(job_groups[job_id])
        self.solution[i:j] = reversed_segment

    def combine(self, other: StochasticSolver, rng: np.random.Generator):
        other = cast(StochasticJobShopSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to combine solution '{self.solution}' with '{other.solution}'"
            )
        n = len(self.solution)
        i, j = sorted(rng.choice(n, size=2, replace=False))
        subseq = self.solution[i:j]
        new_solution = list(subseq)
        for op in other.solution:
            if op not in new_solution:
                new_solution.append(op)
        
        self.solution = new_solution
