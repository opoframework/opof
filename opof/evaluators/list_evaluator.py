import math
from multiprocessing import Process, Queue
from typing import Any, Dict, Generic, List, TypeVar

import torch
from tqdm import tqdm

import opof
from opof.registry import concurrency


def eval_worker(domain, init_queue, job_queue, result_queue):
    # Init.
    planner = domain.create_planner()
    init_queue.put(True)

    while True:
        job = job_queue.get()
        # Sentinal value for signalling termination.
        if job is None:
            break
        result_queue.put(planner(job[0], job[1], job[2]))


Problem = TypeVar("Problem")


class ListEvaluator(Generic[Problem], opof.Evaluator[Problem]):
    """
    :class:`ListEvaluator` represents an evaluator that evaluates generators against a fixed list
    of problem instances.

    For each problem instance in the list, the generator is called to produce corresponding
    planning parameters. The problem instances and respective planning parameters are run
    against the domain's planner in parallel, and the average result is returned. The number
    of parallel workers respect the ``OPOF_CONCURRENCY`` environment variable (see :ref:`configuring OPOF <configure>`).
    """

    problems: List[Problem]
    workers: List[Process]
    job_queue: Queue
    result_queue: Queue
    terminate: Any

    def __init__(
        self,
        domain: opof.Domain,
        problems: List[Problem],
    ):
        """
        Creates a :class:`ListEvaluator` using a fixed list of problem instances.

        :param domain: Domain
        :param problems: List of problem instances
        """
        self.problems = problems
        init_queue = Queue()
        self.job_queue = Queue()
        self.result_queue = Queue()

        self.workers = []
        num_workers = concurrency()
        for _ in range(num_workers):
            process = Process(
                target=eval_worker,
                args=(domain, init_queue, self.job_queue, self.result_queue),
                daemon=True,
            )
            process.start()
            self.workers.append(process)
        # Wait init.
        for _ in range(num_workers):
            init_queue.get()

    def __del__(self):
        for _ in self.workers:
            self.job_queue.put(None)

    def __call__(
        self,
        generator: opof.Generator[Problem],
    ) -> Dict[str, float]:
        with torch.no_grad():
            # Precompute list in case worker times-out in between argument generation.
            for problem in self.problems:
                (parameters, _, extras) = generator([problem])
                parameters = [p[0].detach().cpu().numpy() for p in parameters]
                self.job_queue.put((problem, parameters, extras))

        # Wait for completion.
        keys = set()
        t = dict()
        c = dict()
        for _ in tqdm(range(len(self.problems)), desc="Evaluating..."):
            result = self.result_queue.get()
            for k in result:
                if (
                    isinstance(result[k], float)
                    or isinstance(result[k], int)
                    or isinstance(result[k], bool)
                ):
                    keys.add(k)
                    if not math.isnan(result[k]):
                        if k not in t:
                            t[k] = 0.0
                        if k not in c:
                            c[k] = 0.0
                        t[k] += result[k]
                        c[k] += 1
        for k in t.keys():
            t[k] /= c[k]
        for k in keys:
            if k not in t:
                t[k] = float("nan")

        return t
