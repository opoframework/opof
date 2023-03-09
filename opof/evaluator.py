from abc import ABCMeta, abstractmethod
from typing import Dict, Generic, TypeVar

from .generator import Generator

Problem = TypeVar("Problem")


class Evaluator(Generic[Problem], metaclass=ABCMeta):
    """
    :class:`Evaluator` is the abstract class representing an evaluator. An evaluator evaluates planning
    performance, along with other optional metrics, for a given generator.
    """

    @abstractmethod
    def __call__(self, generator: Generator[Problem]) -> Dict[str, float]:
        """
        Evaluates and returns the planning performance, along with other optional metrics, for a given generator.

        :param generator: Generator
        :return: Dictionary consisting planning performance and other optional metrics. The ``\"objective\"`` key **must** be present as a :class:`float` value.
        """
