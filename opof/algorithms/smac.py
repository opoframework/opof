import os
import shutil
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Optional

import torch
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Callback, HyperparameterOptimizationFacade, Scenario
from torch.utils.tensorboard import SummaryWriter

from opof.algorithm import Algorithm

from ..domain import Domain
from ..models import StaticGenerator
from ..registry import concurrency


def smac_worker(domain: Domain, job_queue: Queue, result_queue: Queue):
    # Create planner.
    planner = domain.create_planner()

    while True:
        job = job_queue.get()
        # Sentinal value for signalling termination.
        if job is None:
            break
        result = planner(job[0], job[1], job[2])
        result_queue.put((job[0], job[1], result))


class SMAC(Algorithm):
    """
    :class:`SMAC` is a wrapper around the `SMAC3 package <https://github.com/automl/SMAC3>`_,
    an actively maintained tool for algorithm configuration using the latest Bayesian optimization techniques.
    We use the `HPOFacade <https://automl.github.io/SMAC3/main/api/smac.facade.hyperparameter_optimization_facade.html#smac.facade.hyperparameter_optimization_facade.HyperparameterOptimizationFacade>`_
    strategy provided by SMAC3.
    In the context of the planner optimization problem, SMAC learns only a generator :math:`G_\\theta(c) = \\theta` that is unconditional
    (i.e. does not change with the problem instance) and deterministic (i.e., always returns the same planning parameters).
    While SMAC does not exploit information specific to each problem instance, it provides an approach that has strong
    theoretical grounding and serves as a reasonable baseline and sanity check.
    Note that in our implementation, each SMAC training iteration involves doing :math:`batch\\_size` planner calls on :math:`batch\\_size`
    sampled problem instances, with the average performancec returned to SMAC.
    Hence, the number of training iterations should be adjusted accordingly to ensure the same amount of data (in terms of planner calls)
    is used for comparison.
    """

    iterations: int
    batch_size: int

    eval_interval: int

    def __init__(
        self,
        domain: Domain,
        iterations: int,
        batch_size: int = 50,
        eval_interval: int = 20,
        eval_folder: Optional[str] = None,
        save_folder: Optional[str] = None,
    ):
        """
        Constructs an instance of the algorithm for a given domain.

        :param domain: Domain
        :param iterations: Number of training iterations to run before terminating.
        :param batch_size: Number of problem instances to evaluate the query planning parameters on per training iteration.
        :param eval_interval: Interval to evaluate generator.
        :param eval_folder: Optional path to write evaluation logs across training. If :code:`None`, evaluations are not logged.
        :param save_folder: Optional path to save models across training. If :code:`None`, models are not saved.
        """
        super(SMAC, self).__init__(domain, eval_folder, save_folder)
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
                target=smac_worker,
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

        # Prepare SMAC.
        n_hyperparams = sum(
            p.trans_num_inputs for p in self.domain.composite_parameter_space()
        )
        cspace = ConfigurationSpace()
        cspace.add_hyperparameters(
            [Float(f"x{i}", (0, 1)) for i in range(n_hyperparams)]
        )
        scenario = Scenario(
            cspace,
            deterministic=False,
            n_trials=self.iterations + 1,  # First iteration doesn't count.
            # Disable logging. Unfortunately the fix at
            # https://github.com/automl/SMAC3/issues/357 to totally disable this
            # appears to have stopped working in SMAC3 v2.0.0a2, so we use
            # /tmp for now.
            output_directory=Path("/tmp"),
        )

        problems = self.domain.create_problem_set()

        iterations = -1
        has_initial_eval = False
        p_self = self

        def eval_and_log(smbo, it):
            incumbent = smbo.intensifier.get_incumbent()
            if incumbent is None:
                return

            params_flat = [incumbent[f"x{i}"] for i in range(n_hyperparams)]
            params = []
            extras = []
            counter = 0
            for pspace in p_self.domain.composite_parameter_space():
                (p, o) = pspace.trans_forward(
                    torch.tensor(
                        [params_flat[counter : counter + pspace.trans_num_inputs]],
                        dtype=torch.double,
                    )
                )
                params.append(p[0])
                extras.extend(o)
                counter += pspace.trans_num_inputs
            result = p_self.evaluator(StaticGenerator(params, extras))
            for k in result:
                print(f" {k} = {result[k]}")
            if logger is not None:
                for k in result.keys():
                    logger.add_scalar(k, result[k], it)

        class cb(Callback):
            def on_tell_end(self, smbo, info, value):
                nonlocal iterations
                nonlocal p_self
                nonlocal has_initial_eval

                iterations += 1
                print(iterations)

                if (
                    not has_initial_eval
                    and smbo.intensifier.get_incumbent() is not None
                ):
                    has_initial_eval = True
                    eval_and_log(smbo, 0)
                elif iterations > 0 and iterations % p_self.eval_interval == 0:
                    eval_and_log(smbo, iterations * p_self.batch_size)

                if iterations >= p_self.iterations:
                    return False

        def obj(x: Configuration, seed: int):

            # Create jobs.
            params_flat = [x[f"x{i}"] for i in range(n_hyperparams)]
            params = []
            extras = []
            counter = 0
            for pspace in p_self.domain.composite_parameter_space():
                (p, o) = pspace.trans_forward(
                    torch.tensor(
                        [params_flat[counter : counter + pspace.trans_num_inputs]],
                        dtype=torch.double,
                    )
                )
                params.append(p[0])
                extras.extend(o)
                counter += pspace.trans_num_inputs

            for _ in range(self.batch_size):
                problem = problems()
                job_queue.put(
                    (problem, [p.detach().cpu().numpy() for p in params], extras)
                )

            # Wait complete.
            results = [result_queue.get() for _ in range(self.batch_size)]
            value = sum(result[2]["objective"] for result in results) / self.batch_size

            # We want to maximize, but SMAC minimizes. So we use the negative value.
            return -value

        facade = HyperparameterOptimizationFacade(
            scenario, obj, callbacks=[cb()], overwrite=True
        )
        facade.optimize()
