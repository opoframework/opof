import itertools
import os
from multiprocessing import Process, Queue
from typing import Any, List, Optional, TypeVar

import numpy as np
import torch
import torch.nn.utils
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from ..algorithm import Algorithm
from ..domain import Domain
from ..evaluator import Evaluator
from ..models import FCResNetCritic, FCResNetGenerator
from ..registry import concurrency

LOG_ALPHA_INIT = -1.0
LOG_ALPHA_MIN = -50.0
LOG_ALPHA_MAX = 20.0


def gc_worker(
    domain: Domain,
    init_queue: Queue,
    request_queue: Queue,
    job_queue: Queue,
    result_queue: Queue,
):
    # Init.
    planner = domain.create_planner()
    problems = domain.create_problem_set()
    init_queue.put(True)

    # Create problem.
    problem = problems()

    while True:
        request_queue.put(problem)
        job = job_queue.get()
        # Sentinal value for signalling termination.
        if job is None:
            break
        result = planner(job[0], job[1], job[2])
        result_queue.put((job[0], job[1], result))
        problem = problems((job[0], job[1], result))


def is_valid(x):
    return not torch.any(torch.isnan(x).logical_or(torch.isinf(x)))


Problem = TypeVar("Problem")


class GC(Algorithm):
    """
    :class:`GC` is our implementation of the Generator-Critic algorithm introduced across the recent works of [Lee2021a]_, [Lee2022a]_, and [Danesh2022a]_.

    Two neural networks are learned simultaneously -- a *generator network* representing the generator :math:`G_\\theta(c)` and a *critic network*.
    The generator is stochastic, and maps a problem instance :math:`c \\in \\mathcal{C}` to a sample of planning parameters :math:`x \\in \\mathcal{X}`.
    The critic maps :math:`c` and :math:`x` into a real distribution modeling :math:`\\boldsymbol{f}(x; c)`.

    During training, the stochastic generator takes a random problem instance :math:`c \\in \\mathcal{C}` and produces a sample :math:`x \\in \\mathcal{X}`,
    which is used to probe the planner. The planner's response, along with :math:`c` and :math:`x`, are used to update the critic via supervision loss.
    The critic then acts as a *differentiable surrogate objective* for :math:`\\boldsymbol{f}(x; c)`, passing gradients to the generator via the chain rule.
    """

    device: torch.device
    dtype: torch.dtype

    num_workers: int
    lr: float
    alpha_lr: float
    max_buffer_size: int
    min_buffer_size: int
    batch_size: int

    iterations: int

    eval_interval: int
    evaluator: Evaluator
    save_interval: int

    def __init__(
        self,
        domain: Domain,
        iterations: int,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        dtype: torch.dtype = torch.float32,
        lr: float = 1e-4,
        alpha_lr: Optional[float] = None,
        min_buffer_size: int = 1000,
        max_buffer_size: Optional[int] = 100000,
        batch_size: int = 64,
        eval_interval: int = 1000,
        save_interval: Optional[int] = None,
        eval_folder: Optional[str] = None,
        save_folder: Optional[str] = None,
    ):
        """
        Constructs an instance of the algorithm for a given domain.

        :param domain: Domain
        :param iterations: Number of training iterations to run before terminating.
        :param device: Device to train models on.
        :param dtype: Type to use for models.
        :param lr: Learning rate for models.
        :param alpha_lr: Learning rate for entropy regularization term. If :code:`None`, the value of :code:`3 * lr` is used.
        :param min_buffer_size: Minimum size of replay buffer to wait for before updating models.
        :param max_buffer_size: Maximum size of replay buffer before evicting old items.
        :param batch_size: Number of items to sample from the replay buffer per training iteration.
        :param eval_interval: Interval to evaluate generator.
        :param save_interval: Interval to save models. If :code:`None`, the value of :code:`iterations / 100` is used.
        :param eval_folder: Optional path to write evaluation logs across training. If :code:`None`, evaluations are not logged.
        :param save_folder: Optional path to save models across training. If :code:`None`, models are not saved.
        """
        super(GC, self).__init__(domain, eval_folder, save_folder)
        self.iterations = iterations

        self.device = device
        self.dtype = dtype

        self.num_workers = concurrency()
        self.lr = lr
        self.alpha_lr = alpha_lr if alpha_lr is not None else 3 * self.lr
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = (
            max_buffer_size if max_buffer_size is not None else self.iterations
        )
        self.batch_size = batch_size

        self.eval_interval = eval_interval
        self.evaluator = self.domain.create_evaluator()
        self.save_interval = (
            save_interval if save_interval is not None else int(self.iterations / 100)
        )

    def __call__(self):
        # Launch workers.
        init_queue: Queue = Queue()
        request_queue: Queue = Queue()
        job_queue: Queue = Queue()
        result_queue: Queue = Queue()
        workers = [
            Process(
                target=gc_worker,
                args=(self.domain, init_queue, request_queue, job_queue, result_queue),
                daemon=True,
            )
            for _ in range(self.num_workers)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            init_queue.get()

        # Logger.
        logger: Optional[SummaryWriter] = None
        if self.eval_folder is not None:
            logger = SummaryWriter(self.eval_folder)

        try:
            # Replay buffer.
            buffer = []

            # Models.
            generator = FCResNetGenerator(self.domain).to(self.device, self.dtype)
            critic = FCResNetCritic(self.domain).to(self.device, self.dtype)
            log_alpha: Optional[torch.Tensor] = None
            entropy_target: Optional[torch.Tensor] = None

            # Optimizers.
            critic_optim = Adam(critic.parameters(), lr=self.lr)
            generator_optim = Adam(generator.parameters(), lr=self.lr)
            alpha_optim: Optional[torch.optim.Optimizer] = None

            # Training loop.
            iteration = 0
            while iteration < self.iterations:

                # Ensure that results have been processed before adding more jobs.
                while result_queue.qsize() > 0:
                    # Get result.
                    job_result = result_queue.get()

                    # Append to buffer.
                    if job_result is not None:
                        buffer.append(
                            (
                                job_result[0],
                                [torch.tensor(p) for p in job_result[1]],
                                job_result[-1]["objective"],
                            )
                        )
                        buffer = buffer[-self.max_buffer_size :]

                    # Train.
                    if len(buffer) < self.min_buffer_size:
                        print("\033[K", end="")
                        print("Buffer", len(buffer), end="\r")
                        continue

                    # Sample problem.
                    batch = [
                        buffer[i]
                        for i in torch.randint(0, len(buffer), (self.batch_size,))
                    ]
                    problem = [x[0] for x in batch]
                    parameters = [x[1] for x in batch]
                    obj = torch.tensor([x[2] for x in batch], device=self.device)

                    # Update critic.
                    critic.train()
                    critic_accuracy = (
                        critic(
                            problem,
                            [
                                torch.stack(p).to(self.dtype).to(self.device)
                                for p in zip(*parameters)
                            ],
                        )
                        .log_prob(obj)
                        .mean()
                    )
                    debug_critic_accuracy = critic_accuracy.detach().item()
                    if is_valid(critic_accuracy):
                        critic_optim.zero_grad()
                        (-critic_accuracy).backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                        for p in critic.parameters():
                            if p is not None and p.grad is not None:
                                torch.nan_to_num(p.grad)
                        critic_optim.step()
                        for p in critic.parameters():
                            if p is not None:
                                torch.nan_to_num(p)

                    # Update generator.
                    critic.eval()
                    generator.train()
                    (parameters, entropy, _) = generator(problem)
                    # Infer log_alpha dimensions lazily from entropy.
                    if log_alpha is None:
                        log_alpha = torch.tensor(
                            np.array(
                                [LOG_ALPHA_INIT] * entropy[0].reshape(-1).shape[0]
                            ),
                            requires_grad=True,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        entropy_target = torch.tensor(
                            list(
                                itertools.chain(
                                    *[
                                        space.dist_target_entropy
                                        for space in self.domain.composite_parameter_space()
                                    ]
                                )
                            ),
                            requires_grad=False,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        alpha_optim = torch.optim.Adam([log_alpha], lr=self.alpha_lr)
                    entropy = entropy.reshape(-1, log_alpha.shape[0])
                    generator_performance = critic(problem, parameters).mean
                    dual = (log_alpha.exp().detach() * entropy).sum(dim=-1)
                    debug_generator_perf = generator_performance.mean().detach().item()
                    debug_entropy = entropy.mean().detach().item()
                    if is_valid(generator_performance) and is_valid(dual):
                        generator_optim.zero_grad()
                        # We want to maximize performance and dual, but torch does
                        # minimization. So we use the negative.
                        (-generator_performance - dual).mean().backward()
                        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                        for p in generator.parameters():
                            if p is not None and p.grad is not None:
                                torch.nan_to_num(p.grad)
                        generator_optim.step()
                        for p in generator.parameters():
                            if p is not None:
                                torch.nan_to_num(p)

                    # Update entropies.
                    # Update log alpha.
                    alpha_optim.zero_grad()
                    alpha_loss = log_alpha * ((entropy - entropy_target).detach())
                    alpha_loss.mean().backward()
                    assert log_alpha.grad is not None
                    with torch.no_grad():
                        log_alpha.grad *= (
                            ((-log_alpha.grad >= 0) | (log_alpha >= LOG_ALPHA_MIN))
                            & ((-log_alpha.grad < 0) | (log_alpha <= LOG_ALPHA_MAX))
                        ).type(self.dtype)
                    alpha_optim.step()
                    debug_log_alpha = log_alpha.mean().item()

                    debug_recent = sum(r[2] for r in buffer[-100:]) / 100

                    # Evaluate.
                    it = iteration if iteration == 0 else iteration + 1
                    if it % self.eval_interval == 0:
                        generator.eval()
                        result = self.evaluator(generator)
                        print(
                            f"Step = {iteration if iteration == 0 else iteration + 1}"
                        )
                        for k in result.keys():
                            print(f"  {k} = {result[k]}")
                            # Log eval.
                            if logger is not None:
                                logger.add_scalar(k, result[k], it)

                    # Save.
                    if self.save_folder is not None and it % self.save_interval == 0:
                        os.makedirs(self.save_folder, exist_ok=True)
                        torch.save(
                            generator.state_dict(),
                            f"{self.save_folder}/generator.{it:07d}.pt",
                        )
                        torch.save(
                            critic.state_dict(),
                            f"{self.save_folder}/critic.{it:07d}.pt",
                        )

                    # Logging.
                    iteration += 1
                    debug: List[Any] = []
                    debug.append(iteration)
                    debug.append(debug_critic_accuracy)
                    debug.append(debug_generator_perf)
                    debug.append(debug_entropy)
                    debug.append(debug_log_alpha)
                    debug.append(debug_recent)
                    if (
                        self.eval_interval is None
                        or iteration % max(1, int(self.eval_interval / 100)) == 0
                    ):
                        print("\033[K", end="")
                        print(
                            *[str(v)[:7].rjust(7, " ") + "  " for v in debug],
                            end="\r",
                        )

                # Respond to pending requests only after results have cleared.
                problem = []
                while request_queue.qsize() > 0:
                    # Get request.
                    problem.append(request_queue.get())
                if len(problem) > 0:
                    generator.eval()
                    (parameters, _, extras) = generator(problem)
                    # Submit jobs.
                    for (i, _problem) in enumerate(problem):
                        _parameters = [p[i].detach().cpu().numpy() for p in parameters]
                        job_queue.put((_problem, _parameters, extras))
        finally:
            for _ in workers:
                job_queue.put(None)
