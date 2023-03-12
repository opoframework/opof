from abc import ABCMeta, abstractmethod
from typing import Any, List, Tuple

from torch import Tensor
from torch.nn import Module


class ParameterSpace(metaclass=ABCMeta):
    """
    :class:`ParameterSpace` is the abstract base class representing a parameter space.
    """

    @abstractmethod
    def rand(self, batch_size: int) -> Tensor:
        """
        Returns a batch of uniformly-distributed samples from the parameter space.

        :param inputs: Batch size
        :return: Samples
        """

    @property
    @abstractmethod
    def trans_num_inputs(self) -> int:
        """
        Returns the required number of points in :math:`[0, 1]` to be 
        transformed into a single point in the parameter space.

        :return: Required number of input points
        """

    @abstractmethod
    def trans_forward(self, inputs: Tensor) -> Tuple[Tensor, List[Any]]:
        """
        Transforms a set of points in :math:`[0, 1]` into a single point in the parameter space. 
        The transformation may also return extra objects which are passed to the planner.

        :param inputs: Input points
        :return: Parameter values and extra objects
        """

    @property
    @abstractmethod
    def dist_num_inputs(self) -> int:
        """
        Returns the required number of raw parameters by a distribution
        over the parameter space.

        :return: Required number of raw parameters
        """

    @property
    @abstractmethod
    def dist_target_entropy(self) -> List[float]:
        """
        Returns the recommended target entropy for a distribution
        over the parameter space.

        :return: Target entropy
        """

    @abstractmethod
    def create_sampler(self) -> Module:
        """
        Creates and returns a :class:`torch.nn.Module` which transforms
        raw distribution parameters into distribution samples and entropies
        in the parameter space. The sampler may also return extra objects
        which are passed to the planner.

        :return: Sampler module
        """

    @abstractmethod
    def create_embedding(self) -> Module:
        """
        Creates and returns a :class:`torch.nn.Module` which transforms
        parameters into real vectors of fixed length.

        :return: Embedding module
        """
