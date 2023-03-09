.. _create domain:

Create Your Own Domain
========================

In this tutorial, we look at how you can specify your own domain for use in OPOF.

Random Walk Optimization
------------------------

We use the example of a `random walk` planner optimization problem. In a 2D grid world, an
agent wants to move from a start position to a goal position, avoiding randomly spawned obstacles.
At each step, the agent randomly selects a direction to move,
where the direction probabilities are fixed across all steps. The agent wants to reach the
goal in as few steps as possible. Given a randomly selected problem instance
(defined by the start, goal, and obstacle positions), can we determine the best direction
probabilities to use?

Domain Overview
---------------

To specify the random walk planner optimization problem, we need to implement a standard set of methods,
which collectively form a `domain`. Such a domain implements four key methods in the :class:`opof.Domain`
abstract class:

.. code-block:: python
  :linenos:

  import opof
  import torch
  from typing import List

  class RandomWalk(opof.Domain):

      def create_problem_set() -> opof.ProblemSet:
          ...

      def composite_parameter_space() -> List[opof.ParameterSpace]:
          ...

      def create_planner() -> opof.Planner:
          ...

      def create_problem_embedding() -> torch.nn.Module:
          ...

Specifying the Problem Set
--------------------------

We first specify the distribution of problem instances by implementing :meth:`create_problem_set`.
We pre-generate a list of 1000 problem instances, where the start, goal, and obstacle
positions are uniformly selected and ensured to be distinct. We fix the board size to :math:`10 \times 10`,
and use :math:`20` obstacles.


.. code-block:: python
  :linenos:

  ...

  class RandomWalk(opof.Domain):

      def create_problem_set(self) -> opof.ProblemSet:
          # Helper method to sample position on a board.
          def rand_pos():
              return (np.random.randint(0, 10), np.random.randint(0, 10))

          # Sample 1000 problems of (board, start, goal) where obstacles,
          # start, and goal do not overlap.
          problems = []
          for _ in range(1000):
              board = np.zeros((10, 10), dtype=np.uint8)
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
          return opof.problem_sets.ProblemList(problems)

      ...

      ...

      ...

We can't return the list of problem instances immediately, as OPOF expects a distribution (:class:`opof.ProblemSet`).
The return statement (line 27) wraps and returns the list as an instance of
:class:`opof.problem_sets.ProblemList`, a built-in implementation of :class:`opof.ProblemSet` which
shuffles and cycles through the fixed list of problem instances repeatedly.

Specifying the Parameter Space
------------------------------

Next, we need to specify the parameter space of our domain by implementing :meth:`composite_parameter_space`.
Our planner requires the probabilities of moving in each direction as parameters. This is a single real vector
whose entries are non-negative and sum to :math:`1` -- in other words, |simplex vectors|.

.. code-block:: python
  :linenos:

  ...

  class RandomWalk(opof.Domain):

      ...

      def composite_parameter_space() -> List[opof.ParameterSpace]:
          return [opof.parameter_spaces.Simplex(1, 4)]

      ...

      ...

The :meth:`composite_parameter_space` method returns a list of individual parameter spaces whose union
make up the problem's parameter space. Here, we specify that we have one such parameter space --
a :class:`opof.parameter_spaces.Simplex` which holds a single simplex vector with 4 variables.

.. note::

   In OPOF, most built-in :ref:`parameter spaces <parameter spaces>` start with a ``count`` argument, which
   lets you hold multiple of the same type of parameter in a single parameter space. This is for efficiency
   purposes, since the generation of parameters are batched per parameter space.

   For example, ``[Simplex(1000, 4)]`` is equivalent to ``[Simplex(1, 4) for _ in range(1000)]``. The former
   batches the generation of the :math:`1000` vectors into a single call, but the latter does :math:`1000`
   individual calls, making it much less efficient.

.. |simplex vectors| raw:: html

   <em><a target="blank_" href="https://en.wikipedia.org/wiki/Simplex">simplex vector</a></em>

Specifying the Planner
----------------------

Now, we specify the `planner`, which does the random walk for a given problem instance
with the given direction probabilities. If the agent attempts to move "into" an obstacle or out of
bounds, we treat the agent as having taken a step, but discard the actual movement. A maximum of
:math:`100` steps is allowed for each random walk.

.. code-block:: python
  :linenos:

  ...

  class RandomWalk(Domain):

      ...

      ...

      def create_planner(self) -> opof.Planner:
          class RandomWalkPlanner(opof.Planner):
              def __call__(self, problem, parameters):
                  # Extract problem information.
                  (board, start, goal) = problem
                  # Extract parameters.
                  probs = parameters[0][0]

                  # Run random walk.
                  pos = start
                  steps = 0
                  while steps <= 100:
                      # Compute next position.
                      action = np.random.choice(4, p=probs)
                      action = [(1, 0), (-1, 0), (0, 1), (0, -1)][action]
                      next_pos = (pos[0] + action[0], pos[1] + action[1])

                      # Move only if valid.
                      if not (
                          pos[0] < 0 or pos[0] >= board.shape[0]
                          or pos[1] < 0 or pos[1] >= board.shape[1]
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

           return RandomWalkPlanner()

      ...

Note how the parameters are unpacked (line 11): since our domain's composite parameter space is a ``[Simplex(1, 4)]``,
the planner expects a nested list of values which looks like ``[[[0.1, 0.3, 0.2, 0.3]]]``. The inner
``[[0.1, 0.3, 0.2, 0.3]]`` corresponds to the one entry of the ``Simplex(1, 4)`` parameter space.

.. note::

   To better illustrate the parameters passed to the planner: if, for example, we had a
   ``[Simplex(2, 2), Simplex(1, 3)]`` composite parameter space,
   then the planner would see something like ``[[[0.7, 0.3], [0.2, 0.8]], [[0.1, 0.4, 0.5]]]``.

OPOF expects the planner to return (line 39) a dictionary of values. Furthermore, the ``"objective"`` entry
must be present with a float value, which corresponds to the sample of the
planning objective :math:`\boldsymbol{f}(x; c)`. Other metrics and objects may be present to track additional key statistics and results of the planner run.

Specifying the Problem Embedding
--------------------------------

In order to use generators :math:`G_\theta(c)` which transforms problem instances into parameters,
we need a way to `embed` the problem instances into a consistent format. This is done by implementing
:meth:`create_problem_embedding()` to return a :class:`torch.nn.Module`.

For our domain, we return a :class:`torch.nn.Module` which takes a list of problem instances, merges
the start, goal, and obstacle information into a single real vector, and combines the result across the
problem instances.


.. code-block:: python
  :linenos:

  ...

  class RandomWalk(Domain):

      ...

      ...

      ...

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

Note the use of the ``dummy_param`` to track which device and dtype the problem instances convert to.
This is necessary since we do not know where or how the :class:`torch.nn.Module` is being used.
The ``dummy_param`` maintains these information implicitly.

Train the Domain
----------------

We have now specified the key requirements of a :class:`opof.Domain`, and are ready to train it.
We import the built-in :class:`opof.algorithms.GC` algorithm and run it as such:

.. code-block:: python
   :linenos:

   from opof.algorithms import GC

   # Create domain instance.
   domain = RandomWalk()

   # Create GC algorithm instance.
   algorithm = GC(domain, iterations=10000)

   # Run training.
   algorithm()

