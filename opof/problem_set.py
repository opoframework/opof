from abc import ABCMeta, abstractmethod
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

from numpy import ndarray

Problem = TypeVar("Problem")


class ProblemSet(Generic[Problem], metaclass=ABCMeta):
    """
    :class:`ProblemSet` is the abstract base class representing distributions :math:`\\mathcal{D}`
    over problem instances :math:`c \\in \\mathcal{C}`.
    """

    @abstractmethod
    def __call__(
        self, previous: Optional[Tuple[Problem, List[ndarray], Dict[str, float]]] = None
    ) -> Problem:
        """
        Returns a problem instance from the problem set.

        :param previous: An optional value that the problem set may use before returning a problem instance. It consists of a prior problem instance, planning parameters, and results from a corresponding planner call.
        :return: Problem instance
        """
