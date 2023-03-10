.. raw:: html

    <style>
    #home > h1 {
      display: none;
    }
    </style>

`Home`
==============

.. image:: _static/img/banner.svg
  :width: 400
  :align: center

OPOF, the Open-Source Planner Optimization Framework, is an open source 
framework for developing domains and algorithms for planner optimization. 
It provides a standard API to communicate between optimization algorithms 
and domains, along with a set of stable algorithm implementations.

OPOF is developed and maintained by the `Kavraki Lab <https://www.kavrakilab.org/>`_ at Rice University.

Algorithms
----------

OPOF includes the following stable algorithm implementations.

-  `Generator-Critic
   (GC) <https://opof.kavrakilab.org/algorithms/GC.html>`__ - Learns a
   conditional and stochastic generator using gradient-based deep
   learning techniques.
-  `SMAC <https://opof.kavrakilab.org/algorithms/SMAC.html>`__ - Learns
   an unconditional and deterministic generator using the latest
   Bayesian optimization techniques. Wrapper around the actively
   maintained `SMAC3 <https://github.com/automl/SMAC3>`__ Bayesian
   optimization library.

We expect the list to grow with time, and welcome any additions.

Domains
-------

Domains are available as external packages maintained separately from
OPOF. Some existing domain include:

-  `opof-grid2d <https://github.com/opoframework/opof-grid2d>`__ -
   Simple navigation domains in a 2D grid world to help users
   familiarize with OPOF. They also act as a sanity check for developing
   optimization algorithms.
-  `opof-sbmp <https://github.com/opoframework/opof-sbmp>`__ -
   Sampling-based motion planning (SBMP) domains for high-DoF robots to
   accomplish real-world picking tasks. They include the optimization of
   planner hyperparameters, sampling distributions, and projections.
-  `opof-pomdp <https://github.com/opoframework/opof-pomdp>`__ - Online
   POMDP planning domains for 2D navigation under uncertainty. They
   include the optimization of macro-actions.

We expect the list to grow with time, and welcome any additions.

Getting Started
---------------

For installation instructions and API code examples, please refer to 
:ref:`getting started <getting started>`.


Citing
------

TBC

License
-------

OPOF is licensed under the `BSD-3
license <https://github.com/opoframework/opof/blob/master/LICENSE.md>`__.

OPOF is developed and maintained by the `Kavraki Lab <https://www.kavrakilab.org/>`_ at Rice University, funded in part by NSF RI 2008720 and Rice University funds.

.. toctree::
   :caption: User Guide
   :hidden:

   Home <self>
   guides/getting_started
   guides/background

.. toctree::
   :caption: Tutorials
   :hidden:
   
   tutorials/configure
   tutorials/create_domain

.. toctree::
   :maxdepth: 2
   :caption: Common
   :hidden:

   Domains <common/domain>
   common/problem_sets
   common/parameter_spaces
   common/planners
   common/evaluators

.. toctree::
   :maxdepth: 2
   :caption: Algorithms
   :hidden:

   algorithms/base
   GC <algorithms/GC>
   SMAC <algorithms/SMAC>

.. toctree::
   :maxdepth: 2
   :caption: Models
   :hidden:

   models/base_generator
   models/fc_resnet
   models/static
