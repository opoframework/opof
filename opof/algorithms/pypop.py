'''
from opof.algorithms import PyPOP
from opof_grid2d.domains import RandomWalk2DOpt


#below imports taken from gc.py reference - do we need all?
import itertools
import os
from multiprocessing import Process, Queue

import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from opof import Algorithm
from opof import Domain
from opof import Evaluator
'''

from ..algorithm import Algorithm
from ..domain import Domain
from typing import Any, List, Optional, TypeVar
import torch
import torch.nn.utils
from pypop7.optimizers.es.mmes import MMES
import numpy as np

class PyPOP(Algorithm):

    iterations: int
    batch_size: int

    def __init__(self, 
                 domain: Domain, 
                 iterations: int, 
                 batch_size: int,
                eval_folder: Optional[str] = None,
                save_folder: Optional[str] = None):
        
        super(PyPOP, self).__init__(domain, eval_folder, save_folder)

        self.iterations = iterations
        self.batch_size = batch_size

    def x_to_opofparam(self, x: int, domain: Domain):
        x = torch.tensor(x).unsqueeze(0)
        return [domain.composite_parameter_space()[0].trans_forward(x)[0][0].numpy()]

    #figure out how to terminate after n iterations
    #plug in the batch_size
    #derive parameter count from the domain
    #transform pyPOP params into opof params
    def __call__(self):
        problems = self.domain.create_problem_set()
        planner = self.domain.create_planner()
        #num_params = len(self.domain.composite_parameter_space()[0])
        fitness_values = []
        best_fitness = None

        #convert pypop params into opof params and run 50 times (rosenbrock)
        def fn(x : List[float]):
            params = []
            extras = []
            counter = 0
            for ps in self.domain.composite_parameter_space():
                (p, o) = ps.trans_forward(
                    torch.tensor(
                        np.array([x[counter : counter + ps.trans_num_inputs]]),
                        dtype=torch.double,
                    )
                )
                params.append(p[0])
                extras.extend(o)
                counter += ps.trans_num_inputs

            s = 0
            for _ in range(self.batch_size):
                problem = problems()
                result = planner(problem, [p.detach().cpu().numpy() for p in params], extras)
                s += result['objective']
            return -s / self.batch_size

        ndim_problem = sum(ps.trans_num_inputs for ps in self.domain.composite_parameter_space())
        problem = {'fitness_function': fn,  # cost function
                'ndim_problem': ndim_problem,  # dimension
                'lower_boundary': np.zeros((ndim_problem)),  # search boundary
                'upper_boundary': np.ones((ndim_problem,))}

        # define all the necessary algorithm options (which differ among different optimizers)
        options = {'fitness_threshold': -np.inf,  # terminate when the best-so-far fitness is lower than this threshold
                'max_function_evaluations': self.iterations, 
                'sigma': 0.3,  # initial global step-size of search distribution
                'verbose': 1}
        mmes = MMES(problem, options)  # initialize the optimizer
        results = mmes.optimize()  # run its (time-consuming) search process

        '''
        for i in range(self.iterations):
            # confused with this
            #candidates = np.random.uniform(low=0.0, high=1.0, size=(self.batch_size, num_params))
            fitnesses = []

            for j in range(self.batch_size):
                #below doesn't work
                opof_params = x_to_opofparam(candidates[j])
                result = planner(problems(), opof_params, [])
                fitness = result["objective"]
                fitnesses.append(fitness)

                if best_fitness is None or fitness < best_fitness:
                    best_fitness = fitness

            fitness_values.append(fitnesses)

        return best_fitness
        '''


    

"""""
    def test(self):
        print("iterations", self.iterations)
        print("batch size", self.batch_size)
"""""