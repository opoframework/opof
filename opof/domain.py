from abc import ABCMeta, abstractmethod
from typing import Generic, List, TypeVar

from torch import nn

from .evaluator import Evaluator
from .parameter_space import ParameterSpace
from .planner import Planner
from .problem_set import ProblemSet

Problem = TypeVar("Problem")


class Domain(Generic[Problem], metaclass=ABCMeta):
    """
    :class:`Domain` is the base class representing :ref:`domains <domains>`.
    """

    @abstractmethod
    def create_problem_set(
        self,
    ) -> ProblemSet[Problem]:
        """
        Creates and returns a :class:`opof.ProblemSet` representing :math:`\\mathcal{D}`, the
        distribution over problem instances in :math:`c \\in \\mathcal{C}`.

        :return: Problem set
        """
    
    @abstractmethod
    def composite_parameter_space(self) -> List[ParameterSpace]:
        """
        Returns a list of :class:`opof.ParameterSpace` instances that make up :math:`\\mathcal{X}`,
        the space of planning parameters. 

        OPOF provides :ref:`implementations of commonly used
        parameter spaces <parameter spaces>`.

        :return: List of parameter spaces
        """

    @abstractmethod
    def create_planner(self) -> Planner[Problem]:
        """
        Creates and returns a :class:`opof.Planner` representing a planner. It produces samples of the planning
        objective :math:`\\boldsymbol{f}(x; c)`, along with other optional metrics and objects.

        :return: Planner
        """

    @abstractmethod
    def create_problem_embedding(self) -> nn.Module:
        """
        Creates and returns a :class:`torch.nn.Module` which transforms problem instances into
        real vectors of fixed length.

        :return: Embedding module
        """

    def create_evaluator(self) -> Evaluator:
        """
        Creates and returns an :class:`opof.Evaluator` which evaluates planning performance, along with other
        optional metrics, for a given generator.

        By default, this creates a :class:`opof.evaluators.ListEvaluator` using 100 random problem instances
        drawn from the problem set defined in :meth:`create_problem_set`.

        :return: Evaluator
        """
        from .evaluators import ListEvaluator

        problem_set = self.create_problem_set()
        problems = [problem_set() for _ in range(100)]
        return ListEvaluator(self, problems)
