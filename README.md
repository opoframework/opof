<p align="center">
    <img src="https://github.com/opoframework/opof/blob/master/docs/_static/img/banner.svg?raw=true" width="500px"/>
</p>

OPOF, the Open-Source Planner Optimization Framework, is an open source framework for developing domains and algorithms for planner optimization. It provides a standard API to communicate between optimization algorithms and domains, along with a set of stable algorithm implementations. 

Our complete documentation is available at [https://opof.kavrakilab.org](https://opof.kavrakilab.org).

[![Build and Test](https://github.com/opoframework/opof/actions/workflows/build_and_test.yml/badge.svg)](https://github.com/opoframework/opof/actions/workflows/build_and_test.yml)

OPOF is maintained by the [Kavraki Lab](https://kavrakilab.org) at Rice University.

## Algorithms

OPOF includes the following stable algorithm implementations. 

* [Generator-Critic (GC)](https://opof.kavrakilab.org/algorithms/GC.html) - Learns a conditional and stochastic generator using gradient-based deep learning techniques.
* [SMAC](https://opof.kavrakilab.org/algorithms/SMAC.html) - Learns an unconditional and deterministic generator using the latest Bayesian optimization techniques. Wrapper around the actively maintained [SMAC3](https://github.com/automl/SMAC3) Bayesian optimization library. 

We expect the list to grow with time, and welcome any additions.

## Domains

Domains are available as external packages maintained separately from OPOF. Some existing domain include:

* [opof-grid2d](https://github.com/opoframework/opof-grid2d) - Simple navigation domains in a 2D grid world to help users familiarize with OPOF. They also act as a sanity check for developing optimization algorithms.
* [opof-sbmp](https://github.com/opoframework/opof-sbmp) - Sampling-based motion planning (SBMP) domains for high-DoF robots to accomplish real-world picking tasks. They include the optimization of planner hyperparameters, sampling distributions, and projections.
* [opof-pomdp](https://github.com/opoframework/opof-pomdp) - Online POMDP planning domains for 2D navigation under uncertainty. They include the optimization of macro-actions.

We expect the list to grow with time, and welcome any additions.

## Installation

To install OPOF's core library, run `pip install opof`.

External packages containing additional domains and algorithms may be installed alongside OPOF. Please refer to the specific package's setup instructions.

OPOF is officially tested and supported for Python 3.9, 3.10, 3.11 on Linux.

## API
Below is an example of interacting with the `RandomWalk2D[11]` domain. 

```python
from opof_grid2d.domains import RandomWalk2D

domain = RandomWalk2D(11)
problems = domain.create_problem_set()
planner = domain.create_planner()

parameters = [pspace.rand(100).numpy() for pspace in domain.composite_parameter_space()]
for i in range(100):
    result = planner(problems(), [p[i] for p in parameters])
    print(result["objective"])
```

Using our built-in `opof.algorithms.GC` planner optimization algorithm is surprisingly easy.

```python
   from opof_grid2d.domains import RandomWalk2D
   from opof.algorithms import GC

   domain = RandomWalk2D(11)
   algo = GC(domain, iterations=100000, eval_folder="results/RandomWalk2D[11]")
   algo()
```

Evaluation results stored at ``results/RandomWalk2D[11]`` can be viewed by running

```console
  $ tensorboard --logdir=results/
```

Our complete documentation is available at [https://opof.kavrakilab.org](https://opof.kavrakilab.org).

## Citing
If you use OPOF, please cite us with:

```
@article{lee23opof,
  author = {Lee, Yiyuan and Lee, Katie and Cai, Panpan and Hsu, David and Kavraki, Lydia E.},
  title = {The Planner Optimization Problem: Formulations and Frameworks},
  booktitle = {arXiv},
  year = {2023},
  doi = {10.48550/ARXIV.2303.06768},
}
```

## License

OPOF is licensed under the [BSD-3 license](https://github.com/opoframework/opof/blob/master/LICENSE.md).

OPOF is maintained by the [Kavraki Lab](https://www.kavrakilab.org/) at Rice University, funded in part by NSF RI 2008720 and Rice University funds.
