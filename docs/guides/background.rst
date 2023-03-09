Background
============

What's OPOF?
------------

The Open Planner Optimization Framework (OPOF) is a framework for planner optimization. The core library consists of standardized classes to implement a planner optimization problem, along with algorithms to solve them. The goal is to make it easier to apply existing algorithms to interesting domains, and to develop better algorithms using existing domains as benchmarks.

.. _planner optimization problem:

Planner Optimization Problem
----------------------------
A `planner optimization problem` is a :math:`4`-tuple :math:`(\mathcal{C}, \mathcal{X}, \boldsymbol{f}, \mathcal{D})`,
where :math:`\mathcal{C}` is the `problem instance space`, :math:`\mathcal{X}` is the `planning parameter space`, 
:math:`\boldsymbol{f}(x; c)` is the `planning objective`, and :math:`\mathcal{D}` is the `problem instance distribution`.

A `planner` takes as input a problem instance :math:`c \in \mathcal{C}` and planning parameters
:math:`x \in \mathcal{X}`, and performs some computation on :math:`c` using :math:`x`. It returns
a numeric value indicating the quality of the computation, which inherently follows the distribution
:math:`\boldsymbol{f}(x; c)`, the planning objective. The goal is to find a `generator` :math:`G_{\theta}(c)` 
that maps a problem instance :math:`c \in \mathcal{C}` to planning parameters :math:`x \in \mathcal{X}`, 
such that the `expected planning performance` is maximized:

.. math::
   \underset{\theta}{\arg\max} ~ \mathbb{E}_{c \sim \mathcal{D}}[\mathbb{E}_{x \sim G_\theta(c)}[\mathbb{E}[\boldsymbol{f}(x; c)]]]

The generator may be `stochastic`, in which case samples :math:`x \sim G_\theta(c)` are produced whenever
the generator is called. On the other hand, a `deterministic` generator always returns the same value 
:math:`x = G_\theta(c)` for the same problem instance :math:`c`. An `unconditional` generator is one
whose output is unaffected by its inputs :math:`c`.

The problem is challenging because the generator :math:`G_\theta(c)` needs to be learned solely through 
interaction with the planner. In particular, we only have access to the planner which produces samples of
:math:`\boldsymbol{f}(x; c)`. We do not assume to have an analytical, much less differentiable, form for
:math:`\boldsymbol{f}(x; c)`.

History of OPOF
---------------

The planner optimization problem is a relatively new problem formulation. It has recently been used to solve
several important problems in integrating planning and learning, especially in robotics.

In its original inception, it was used to learn macro-actions for online POMDP planning [Lee2021a]_, where
the original `generator-critic` algorithm was devised to solve the planner optimization problem.
It was later used to learning sampling distributions for SBMPs [Lee2022a]_, and to learn attention
for autonomous driving in crowded environments [Danesh2022a]_. Given the generality of the problem formulation, 
we believe that many potential and impactful applications remain to be explored.

While these works have vastly different codebases, they adapt the exact same general problem formulation
and algorithm with near-zero changes. OPOF was thus initiated in an effort to place such works in
a coherent and reusable framework, to improve overall research efforts.

.. [Lee2021a] \  Y. Lee, P. Cai, and D. Hsu. "MAGIC: Learning Macro-Actions for Online POMDP Planning". In `Robotics: Science and Systems (RSS)`, 2021
.. [Lee2022a] \  Y. Lee, C. Chamzas, and L. E. Kavraki. "Adaptive Experience Sampling for Motion Planning using the Generator-Critic Framework". `IEEE Robotics and Automation Letters (RA-L)`, 2022.
.. [Danesh2022a] \  M. H. Danesh, P. Cai, D. Hsu. "LEADER: Learning Attention over Driving Behaviors for Planning under Uncertainty". In `Conference on Robot Learning (CORL)`, 2022.
