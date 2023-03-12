from typing import Any, List, Tuple

import torch
import torch.distributions
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ..parameter_space import ParameterSpace

EPS = 1e-6


class IntervalSampler(Module):
    count: int

    def __init__(self, count: int):
        super(IntervalSampler, self).__init__()
        self.count = count

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, List[Any]]:
        assert len(inputs.shape) == 2, "Invalid inputs"

        inputs = inputs.reshape(-1, self.count, 2)
        a = EPS + F.softplus(inputs[:, :, 0])
        b = EPS + F.softplus(inputs[:, :, 1])
        dist = torch.distributions.Beta(a, b)

        return (dist.rsample().reshape(-1, self.count, 1), dist.entropy(), [])


class Interval(ParameterSpace):
    """
    :class:`Interval` represents a parameter space with :math:`count` intervals with
    values in :math:`(0, 1)`.

    To transform a set of points in :math:`[0, 1]` into a single point in this 
    parameter space, we simply return the input points as output.
    
    The output is guaranteed to be uniformly distributed whenever the input values
    are uniformly distributed.

    A distribution over this parameter space is represented using independent
    Beta distributions for each interval. Constructing such a joint distribution
    requires :math:`count \\times 2` raw parameters as input.
    """

    count: int

    def __init__(self, count: int):
        """
        Creates an :class:`Interval` parameter space with ``count`` intervals.

        :param count: Number of intervals
        """
        if count < 1:
            raise ValueError("At least one value required")
        self.count = count

    def rand(self, batch_size: int) -> Tensor:
        """
        Returns a batch of uniformly-distributed samples from the parameter space.

        For :class:`Interval`, this returns values of shape :math:`(batch, count, 1)`.

        :param inputs: Batch size
        :return: Samples
        """
        return self.trans_forward(torch.rand(batch_size, self.trans_num_inputs))[0]

    @property
    def trans_num_inputs(self) -> int:
        """
        Returns the required number of points in :math:`[0, 1]` to be 
        transformed into a single point in the parameter space.

        For :class:`Interval`, this is equal to :math:`count`.

        :return: Required number of input points
        """
        return self.count

    def trans_forward(self, inputs: Tensor) -> Tuple[Tensor, List[Any]]:
        """
        Transforms a set of points in :math:`[0, 1]` into a single point in the parameter space. 
        The transformation may also return extra objects which are passed to the planner.

        For :class:`Interval`, this takes :math:`(batch, count)` samples and
        returns values of shape :math:`(batch, count, 1)`.

        :param inputs: Input points
        :return: Parameter values and extra objects
        """
        assert len(inputs.shape) == 2
        assert inputs.shape[1] == self.trans_num_inputs
        return (inputs.reshape(-1, self.count, 1), [])

    @property
    def dist_num_inputs(self) -> int:
        """
        Returns the required number of raw parameters by a distribution
        over the parameter space.

        For :class:`Interval`, this is equal to :math:`count \\times 2`.

        :return: Required number of raw parameters
        """
        return self.count * 2

    @property
    def dist_target_entropy(self) -> List[float]:
        """
        Returns the recommended target entropy for a distribution
        over the parameter space.

        For :class:`Interval`, this returns :math:`count` values.

        :return: Target entropy
        """
        return [-3.0] * self.count

    def create_sampler(self) -> Module:
        """
        Creates and returns a :class:`torch.nn.Module` which transforms
        raw distribution parameters into distribution samples and entropies
        in the parameter space. The sampler may also return extra objects
        which are passed to the planner.

        For :class:`Interval`, the sampler takes raw distribution parameters of shape
        :math:`(batch, count \\times 2)` and returns a tuple consisting of sampled
        values of shape :math:`(batch, count, 1)` and entropy terms of shape
        :math:`(batch, count)`.

        :return: Sampler module
        """
        return IntervalSampler(self.count)

    def create_embedding(self) -> Module:
        return torch.nn.Flatten(start_dim=1)
