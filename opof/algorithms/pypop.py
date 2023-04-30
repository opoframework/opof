import os
import shutil
from multiprocessing import Process, Queue
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.utils
from pypop7.optimizers.es.mmes import MMES
from torch.utils.tensorboard import SummaryWriter

from ..algorithm import Algorithm
from ..domain import Domain
from ..evaluator import Evaluator
from ..models import StaticGenerator
from ..parameter_space import ParameterSpace
from ..registry import concurrency


def pypop_worker(domain: Domain, job_queue: Queue, result_queue: Queue):
    # Create planner.
    planner = domain.create_planner()

    while True:
        job = job_queue.get()
        # Sentinal value for signalling termination.
        if job is None:
            break
        result = planner(job[0], job[1], job[2])
        result_queue.put((job[0], job[1], result))


def pypop_to_opof(x: List[float], composite_pspace: List[ParameterSpace]):
    # Convert to OPOF params.
    params = []
    extras = []
    counter = 0
    for ps in composite_pspace:
        (p, o) = ps.trans_forward(
            torch.tensor(
                np.clip(np.array([x[counter : counter + ps.trans_num_inputs]]), 0, 1),
                dtype=torch.double,
            )
        )
        params.append(p[0])
        extras.extend(o)
        counter += ps.trans_num_inputs
    return (params, extras)


class PyPop(Algorithm):

    iterations: int
    batch_size: int
    num_workers: int

    eval_interval: int
    evaluator: Evaluator

    def __init__(
        self,
        domain: Domain,
        iterations: int,
        batch_size: int,
        eval_interval: int = 20,
        eval_folder: Optional[str] = None,
        save_folder: Optional[str] = None,
    ):

        super(PyPop, self).__init__(domain, eval_folder, save_folder)

        self.iterations = iterations
        self.batch_size = batch_size
        self.num_workers = concurrency()

        self.eval_interval = eval_interval
        self.evaluator = self.domain.create_evaluator()

    def __call__(self):
        # Prepare workers.
        job_queue: Queue = Queue()
        result_queue: Queue = Queue()
        workers = [
            Process(
                target=pypop_worker,
                args=(self.domain, job_queue, result_queue),
                daemon=True,
            )
            for _ in range(self.num_workers)
        ]
        for worker in workers:
            worker.start()

        # Logger.
        logger: Optional[SummaryWriter] = None
        if self.eval_folder is not None:
            if os.path.exists(self.eval_folder):
                shutil.rmtree(self.eval_folder)
            logger = SummaryWriter(self.eval_folder)

        # Objective function.
        mmes = Union[MMES, None]
        problems = self.domain.create_problem_set()
        has_initial_eval = False

        def fn(x: List[float]):
            nonlocal has_initial_eval

            (params, extras) = pypop_to_opof(x, self.domain.composite_parameter_space())

            # Enqueue to workers.
            for _ in range(self.batch_size):
                problem = problems()
                job_queue.put(
                    (problem, [p.detach().cpu().numpy() for p in params], extras)
                )

            # Wait complete.
            results = [result_queue.get() for _ in range(self.batch_size)]
            value = sum(result[2]["objective"] for result in results) / self.batch_size

            # Evaluate if needed.
            assert mmes is not None
            do_eval = False
            if not has_initial_eval and mmes.best_so_far_x is not None:
                has_initial_eval = True
                do_eval = True
            elif (mmes.n_function_evaluations + 1) % self.eval_interval == 0:
                do_eval = True

            if do_eval:
                (params_eval, extras_eval) = pypop_to_opof(
                    mmes.best_so_far_x, self.domain.composite_parameter_space()
                )
                result = self.evaluator(StaticGenerator(params, extras))
                for k in result:
                    print(f" {k} = {result[k]}")
                if logger is not None:
                    for k in result.keys():
                        logger.add_scalar(
                            k,
                            result[k],
                            (mmes.n_function_evaluations + 1) * self.batch_size,
                        )

            # We want to maximize, but PyPop minimizes. So we use the negative value.
            return -value

        # Setup PyPop.
        ndim_problem = sum(
            ps.trans_num_inputs for ps in self.domain.composite_parameter_space()
        )
        problem = {
            "fitness_function": fn,  # cost function
            "ndim_problem": ndim_problem,  # dimension
            "lower_boundary": np.zeros((ndim_problem)),  # search boundary
            "upper_boundary": np.ones((ndim_problem,)),
        }
        options = {
            "fitness_threshold": -np.inf,
            "max_function_evaluations": self.iterations,
            "sigma": 3.0,
            "verbose": 1,
        }
        mmes = MMES(problem, options)

        # Run.
        mmes.optimize()
