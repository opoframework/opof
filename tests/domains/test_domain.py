import numpy as np
import torch

from opof import Domain
from opof.algorithms import GC, SMAC, PyPop
from opof.parameter_spaces import Simplex
from opof.problem_sets import ProblemList

torch.set_num_threads(1)


class RandomWalk(Domain):
    def __init__(self, size, obstacles, max_steps):
        self.size = size
        self.obstacles = obstacles
        self.max_steps = max_steps

    def create_problem_set(self):
        # Helper method to sample free position on a board.
        def rand_pos():
            return (np.random.randint(0, self.size), np.random.randint(0, self.size))

        # Sample 1000 problems of (board, start, goal) where obstacles,
        # start, and goal do not overlap.
        problems = []
        for _ in range(1000):
            board = np.zeros((self.size, self.size), dtype=np.uint8)
            start = rand_pos()
            while True:
                goal = rand_pos()
                if goal != start:
                    break
            for _ in range(self.obstacles):
                while True:
                    obstacle = rand_pos()
                    if obstacle != start and obstacle != goal and not board[obstacle]:
                        board[obstacle] = 1
                        break
            problems.append((board, start, goal))

        # Return as built-in :class:`ProblemList` problem set.
        return ProblemList(problems)

    def composite_parameter_space(self):
        return [Simplex(1, 4)]

    def create_planner(self):
        def planner(problem, parameters, optionals):
            # Extract problem information.
            (board, start, goal) = problem
            # Extract parameters.
            probs = parameters[0][0]

            # Run random walk.
            pos = start
            steps = 0
            while steps <= self.max_steps:
                # Compute next position.
                action = np.random.choice(4, p=probs)
                action = [(1, 0), (-1, 0), (0, 1), (0, -1)][action]
                next_pos = (pos[0] + action[0], pos[1] + action[1])

                # Move only if valid.
                if not (
                    pos[0] < 0
                    or pos[0] >= self.size
                    or pos[1] < 0
                    or pos[1] >= self.size
                    or board[pos]
                ):
                    pos = next_pos

                # Add to steps.
                steps += 1

                # Check termination
                if pos == goal:
                    break

            # OPOF maximizes objective, but we want to minimze steps.
            # So we add the negative as objective.
            return {"objective": -steps}

        return planner

    def create_problem_embedding(self):
        class MazeEmbedding(torch.nn.Module):
            def __init__(self):
                super(MazeEmbedding, self).__init__()
                self.dummy_param = torch.nn.Parameter(torch.empty(0))

            def forward(self, problems):
                device = self.dummy_param.device
                dtype = self.dummy_param.dtype
                boards = torch.tensor(
                    np.array([p[0] for p in problems]), device=device, dtype=dtype
                )
                boards = boards.flatten(start_dim=1)
                starts = torch.tensor(
                    np.array([p[1] for p in problems]), device=device, dtype=dtype
                )
                goals = torch.tensor(
                    np.array([p[2] for p in problems]), device=device, dtype=dtype
                )
                return torch.concat([boards, starts, goals], dim=-1)

        return MazeEmbedding()


def test_random_walk_domain():
    # Test init.
    domain = RandomWalk(10, 20, 100)

    # Test problem set.
    problems = domain.create_problem_set()
    for _ in range(10):
        problem = problems()
        assert problem[0].sum() == 20

    # Test parameter space creation.
    parameter_spaces = domain.composite_parameter_space()

    # Test planner creation.
    planner = domain.create_planner()
    assert planner is not None

    # Test parameter space sampling.
    parameters = [pspace.rand(10) for pspace in parameter_spaces]
    assert parameters is not None

    # Test planning.
    for i in range(10):
        result = planner(problems(), [p[i].numpy() for p in parameters], [])
        assert "objective" in result


def test_random_walk_gc():
    domain = RandomWalk(10, 20, 100)
    algorithm = GC(domain, 100, min_buffer_size=100, eval_interval=50)
    algorithm()


def test_random_walk_smac():
    domain = RandomWalk(10, 20, 100)
    algorithm = SMAC(domain, 50, 10, eval_interval=25)
    algorithm()


def test_random_walk_pypop():
    domain = RandomWalk(10, 20, 100)
    algorithm = PyPop(domain, 50, 10, eval_interval=25)
    algorithm()
