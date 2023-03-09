from typing import Any, List, Tuple

import torch
import torch.distributions
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from ..parameter_space import ParameterSpace

EPS = 1e-6


class SimplexSampler(Module):
    count: int
    dimension: int

    def __init__(self, count: int, dimension: int):
        super(SimplexSampler, self).__init__()
        self.count = count
        self.dimension = dimension

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, List[Any]]:
        assert len(inputs.shape) == 2, "Invalid inputs"

        inputs = inputs.reshape(-1, self.count, self.dimension)
        x = EPS + F.softplus(inputs)
        dist = torch.distributions.Dirichlet(x)

        return (dist.rsample(), dist.entropy(), [])


class Simplex(ParameterSpace):
    """
    :class:`Simplex` represents a parameter space with :math:`count` vectors on the
    standard simplex in :math:`dimension`-dimensional space. In particular, each
    vector has non-negative entries summing to :math:`1`.

    To transform a set of points in :math:`[0, 1]` into a single point in this 
    parameter space, we utilize the procedure:

    #. For each point, evaluate the inverse CDF of :math:`\\operatorname{Exp}(1)` at that point.
    #. Normalize the result by its sum.

    The output is guaranteed to be uniformly distributed on the standard simplex
    whenever the input values are uniformly distributed.
    (see `CS.StackExchange <https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>`__).

    A distribution over this parameter space is represented using independent
    Dircichlet distributions for each vector. Constructing such a joint
    distribution requires :math:`count \\times dimension` raw parameters
    as input.
    """

    count: int
    dimension: int

    def __init__(self, count: int, dimension: int):
        """
        Creates a :class:`Simplex` parameter space with ``count`` vectors in
        :math:`dimension`-dimensional space.

        :param count: Number of vectors
        :param dimension: Number of dimensions
        """
        if count < 1:
            raise ValueError("At least one value required")
        if dimension < 2:
            raise ValueError("At least two choices required")
        self.count = count
        self.dimension = dimension

    def rand(self, batch_size: int) -> Tensor:
        """
        Returns a batch of uniformly-distributed samples from the parameter space.

        For :class:`Simplex`, this returns values of shape :math:`(batch, count, dimension)`.

        :param inputs: Batch size
        :return: Samples
        """
        return self.trans_forward(torch.rand(batch_size, self.trans_num_inputs))[0]

    @property
    def trans_num_inputs(self) -> int:
        """
        Returns the required number of points in :math:`[0, 1]` to be 
        transformed into a single point in the parameter space.

        For :class:`Simplex`, this is equal to :math:`count \\times dimension`.

        :return: Required number of input points
        """
        return self.count * self.dimension

    def trans_forward(self, inputs: Tensor) -> Tuple[Tensor, List[Any]]:
        """
        Transforms a set of points in :math:`[0, 1]` into a single point in the parameter space. 
        The transformation may also return extra objects which are passed to the planner.

        For :class:`Simplex`, this takes :math:`(batch, count \\times dimension)` samples and
        returns values of shape :math:`(batch, count, dimension)`.

        :param inputs: Input points
        :return: Parameter values and extra objects
        """
        assert len(inputs.shape) == 2
        assert inputs.shape[1] == self.trans_num_inputs
        y = inputs.reshape(-1, self.count, self.dimension)
        y = -torch.log(y)
        return (y / y.sum(dim=-1, keepdim=True), [])

    @property
    def dist_num_inputs(self) -> int:
        """
        Returns the required number of raw parameters by a distribution
        over the parameter space.

        For :class:`Simplex`, this is equal to :math:`count \\times dimension`.

        :return: Required number of raw parameters
        """
        return self.count * self.dimension

    @property
    def dist_target_entropy(self) -> List[float]:
        """
        Returns the recommended target entropy for a distribution
        over the parameter space.

        For :class:`Simplex`, this returns :math:`count` values.

        :return: Target entropy
        """
        return [-5.0 * self.dimension] * self.count

    def create_sampler(self) -> Module:
        """
        Creates and returns a :class:`torch.nn.Module` which transforms
        raw distribution parameters into distribution samples and entropies
        in the parameter space. The sampler may also return extra objects
        which are passed to the planner.

        For :class:`Simplex`, the sampler takes raw distribution parameters of shape
        :math:`(batch, count \\times dimension)` and returns a tuple consisting of sampled
        values of shape :math:`(batch, count, dimension)` and entropy terms of shape
        :math:`(batch, count)`.

        :return: Sampler module
        """
        return SimplexSampler(self.count, self.dimension)

    def create_embedding(self) -> Module:
        return torch.nn.Flatten(start_dim=1)
