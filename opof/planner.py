from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

from numpy import ndarray

Problem = TypeVar("Problem")


class Planner(Generic[Problem], metaclass=ABCMeta):
    """
    :class:`Planner` is the abstract class representing a planner. A planner produces samples of the planning
    objective :math:`\\boldsymbol{f}(x; c)`, along with other optional metrics.
    """

    @abstractmethod
    def __call__(
        self, problem: Problem, parameters: List[ndarray], extras: List[Any]
    ) -> Dict[str, Any]:
        """
        Performs some computation for the given problem instance, planning parameters, 
        and extra objects, and returns a sample of the planning objective, along with 
        other optional metrics and objects.

        :param problem: Problem instance
        :param parameters: Planning parameters
        :param parameters: Optional objects
        :return: Dictionary consisting planning objective and other optional metrics and objects. The ``\"objective\"`` key **must** be present as a :class:`float` value.
        """
