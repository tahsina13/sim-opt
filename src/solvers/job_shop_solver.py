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
    def _build_initial_solution(self) -> list[tuple[int, int]]:
        all_ops = []
        for job_id, job_ops in enumerate(self.jobs):
            for op_idx, op in enumerate(job_ops):
                all_ops.append((job_id, op_idx, op.processing_time))
        
        all_ops.sort(key=lambda x: x[2])
        return [(job_id, op_idx) for job_id, op_idx, _ in all_ops]
    
    def _get_machine_operations(self, machine_id: int) -> list[tuple[int, int]]:
        ops = []
        for job_id, job_ops in enumerate(self.jobs):
            for op_idx, op in enumerate(job_ops):
                if op.machine_id == machine_id:
                    ops.append((job_id, op_idx))
        return ops
    
    def _calculate_ready_times(self, scheduled_machines: set[int]) -> dict[tuple[int, int], float]:
        ready_times = {}
        machine_available = {m: 0.0 for m in scheduled_machines}
        job_available = [0.0] * len(self.jobs)
        
        # for each job calc when operations can start
        for job_id, job_ops in enumerate(self.jobs):
            for op_idx, op in enumerate(job_ops):
                prev_op_time = job_available[job_id]
                
                if op.machine_id in scheduled_machines:
                    machine_time = machine_available[op.machine_id]
                    start_time = max(prev_op_time, machine_time)
                    machine_available[op.machine_id] = start_time + op.processing_time
                else:
                    start_time = prev_op_time
                
                ready_times[(job_id, op_idx)] = start_time
                job_available[job_id] = start_time + op.processing_time
        
        return ready_times
    
    def _sequence_machine(self, machine_id: int, ready_times: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
        machine_ops = self._get_machine_operations(machine_id)
        tail_times = {}
        for job_id, job_ops in enumerate(self.jobs):
            cumulative = 0.0
            for op_idx in range(len(job_ops) - 1, -1, -1):
                op = job_ops[op_idx]
                tail_times[(job_id, op_idx)] = cumulative
                cumulative += op.processing_time
        
        def priority(op_key):
            job_id, op_idx = op_key
            op = self.jobs[job_id][op_idx]
            return ready_times[op_key] + op.processing_time + tail_times[op_key]
        
        machine_ops.sort(key=priority)
        return machine_ops
    
    def _find_bottleneck_machine(self, unscheduled_machines: list[int], scheduled_machines: set[int]) -> int:
        best_machine = unscheduled_machines[0]
        best_makespan = float('inf')
        
        for machine_id in unscheduled_machines:
            test_scheduled = scheduled_machines | {machine_id}
            test_ready_times = self._calculate_ready_times(test_scheduled)
            makespan = 0.0
            for job_id, job_ops in enumerate(self.jobs):
                if len(job_ops) > 0:
                    last_op = (job_id, len(job_ops) - 1)
                    completion = test_ready_times[last_op] + job_ops[-1].processing_time
                    makespan = max(makespan, completion)
            if makespan > best_makespan:  
                best_makespan = makespan
                best_machine = machine_id
        return best_machine
    
    def solve(self):
        tail_times = {}
        for job_id, job_ops in enumerate(self.jobs):
            cumulative = 0.0
            for op_idx in range(len(job_ops) - 1, -1, -1):
                op = job_ops[op_idx]
                tail_times[(job_id, op_idx)] = cumulative
                cumulative += op.processing_time
        job_next_op = [0] * len(self.jobs)
        solution = []
        
        total_ops = sum(len(job) for job in self.jobs)
        
        while len(solution) < total_ops:
            available = []
            for job_id in range(len(self.jobs)):
                if job_next_op[job_id] < len(self.jobs[job_id]):
                    op_idx = job_next_op[job_id]
                    op = self.jobs[job_id][op_idx]
                    priority = op.processing_time + tail_times[(job_id, op_idx)]
                    available.append((priority, job_id, op_idx))
            
            if not available:
                break
            
            available.sort(key=lambda x: x[0], reverse=True)
            _, job_id, op_idx = available[0]
            
            solution.append((job_id, op_idx))
            job_next_op[job_id] += 1
        
        self.solution = solution


class StochasticJobShopSolver(JobShopSolver, StochasticSolver):
    def generate(self, rng: np.random.Generator):
        all_ops = []
        for job_id, job_ops in enumerate(self.jobs):
            for op_idx in range(len(job_ops)):
                all_ops.append((job_id, op_idx))
        indices = rng.permutation(len(all_ops))
        self.solution = [all_ops[i] for i in indices]
    
    def combine(self, other: StochasticSolver, rng: np.random.Generator):
        other = cast(StochasticJobShopSolver, other)
        if self.solution is None or other.solution is None:
            raise RuntimeError(
                f"Failed to combine solution '{self.solution}' with '{other.solution}'"
            )    
        use_parent1 = rng.random(len(self.jobs)) < 0.5 
        parent1_ops = [op for op in self.solution]
        parent2_ops = [op for op in other.solution]
        new_solution = []
        p1_idx = 0
        p2_idx = 0
        
        while p1_idx < len(parent1_ops) or p2_idx < len(parent2_ops):
            if p1_idx < len(parent1_ops):
                job_id, op_idx = parent1_ops[p1_idx]
                if use_parent1[job_id] and (job_id, op_idx) not in new_solution:
                    new_solution.append((job_id, op_idx))
                    p1_idx += 1
                    continue
            if p2_idx < len(parent2_ops):
                job_id, op_idx = parent2_ops[p2_idx]
                if not use_parent1[job_id] and (job_id, op_idx) not in new_solution:
                    new_solution.append((job_id, op_idx))
                    p2_idx += 1
                    continue
            if p1_idx < len(parent1_ops):
                if parent1_ops[p1_idx] not in new_solution:
                    new_solution.append(parent1_ops[p1_idx])
                p1_idx += 1
            elif p2_idx < len(parent2_ops):
                if parent2_ops[p2_idx] not in new_solution:
                    new_solution.append(parent2_ops[p2_idx])
                p2_idx += 1
        self.solution = new_solution
    
    def mutate(self, rng: np.random.Generator):
        if self.solution is None:
            raise RuntimeError(f"Failed to mutate solution '{self.solution}'")
        i = rng.choice(len(self.solution))
        j = rng.choice(len(self.solution))
        self.solution[i], self.solution[j] = self.solution[j], self.solution[i]