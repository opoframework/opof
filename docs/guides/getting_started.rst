.. _getting started:

Getting Started
===============

Installing OPOF
---------------

First, ensure that you have Python >= 3.9. If not, you can use a `conda environment <https://conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_:

.. code-block:: console
  
  $ conda create -n opof python=3.9
  $ conda activate opof

Next, install OPOF's core library and the example domains used in this guide:

.. code-block:: console

  $ pip install opof opof-grid2d


Using Domains
------------------------

OPOF encapsulates planner optimization problems as a ``Domain`` class. It contains all
key components required to specify the problem. Below is an example of interacting
with the ``RandomWalk2D[11]`` domain.

.. code-block:: python

   from opof_grid2d.domains import RandomWalk2D

   domain = RandomWalk2D(11)
   problems = domain.create_problem_set()
   planner = domain.create_planner()

   parameters = [pspace.rand(100).numpy() for pspace in domain.composite_parameter_space()]
   for i in range(100):
       result = planner(problems(), [p[i] for p in parameters], [])
       print(result["objective"])

Training a Generator
--------------------

Using our built-in :class:`opof.algorithms.GC` planner optimization algorithm is surprisingly easy.

.. code-block:: python

   from opof_grid2d.domains import RandomWalk2D
   from opof.algorithms import GC

   domain = RandomWalk2D(11)
   algo = GC(domain, iterations=100000, eval_folder="results/RandomWalk2D[11]")
   algo()

Evaluation results stored at ``results/RandomWalk2D[11]`` can be viewed by running

.. code-block:: console
  
  $ tensorboard --logdir=results/
