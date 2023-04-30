from abc import ABCMeta, abstractmethod
from typing import Any, Generic, List, Optional, Tuple, TypeVar

from torch import Tensor, nn

Problem = TypeVar("Problem")


class Generator(Generic[Problem], nn.Module, metaclass=ABCMeta):
    """
    :class:`Generator` is the abstract base class representing a mapping :math:`G_\\theta`
    from problem instances :math:`c \\in \\mathcal{C}` to distributions over the planning
    parameter space :math:`\\mathcal{X}`.
    """

    @abstractmethod
    def forward(
        self, problem: List[Problem]
    ) -> Tuple[List[Tensor], Optional[Tensor], List[List[Any]]]:
        """
        Transforms a batch of problem instances :math:`c \\in \\mathcal{C}` into a batch
        of distributions over :math:`\\mathcal{X}`, and returns a batch of samples,
        entropies, and extra objects which are passed to the planner.

        :param problem: Batch of problem instances
        :return: Batch of samples, entropies, and extra objects.
        """
