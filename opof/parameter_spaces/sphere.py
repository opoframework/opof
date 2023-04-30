from typing import Any, List, Tuple

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
from power_spherical import PowerSpherical
from torch import Tensor
from torch.nn import Module

from ..parameter_space import ParameterSpace

EPS = 1e-6


class SphereSampler(Module):
    count: int
    dimension: int

    def __init__(self, count: int, dimension: int):
        super(SphereSampler, self).__init__()
        self.count = count
        self.dimension = dimension

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor, List[Any]]:
        assert len(inputs.shape) == 2, "Invalid inputs"

        inputs = inputs.reshape(-1, self.count, self.dimension + 1)
        mean = inputs[:, :, :-1]
        mean = mean / (mean**2).sum(dim=-1, keepdim=True).sqrt()
        concentration = 1 + EPS + F.softplus(inputs[:, :, -1])

        dist = PowerSpherical(mean, concentration)

        return (dist.rsample(), dist.entropy(), [])


class Sphere(ParameterSpace):
    """
    :class:`Sphere` represents a parameter space with :math:`count` unit vectors
    in :math:`dimension`-dimensional space. In particular, each vector has
    :math:`L_2` norm summing to :math:`1`.

    To transform a set of points in :math:`[0, 1]` into a single point in this 
    parameter space, we utilize the procedure:

    #. For each point, evaluate the inverse CDF of :math:`~\\mathcal{N}(0, 1)` at that point.
    #. Normalize the result by its :math:`L_2` norm.

    The output is guaranteed to be uniformly distributed on the unit sphere
    whenever the input values are uniformly distributed.
    (see `CS.StackExchange <https://mathoverflow.net/questions/24688/efficiently-sampling-points-uniformly-from-the-surface-of-an-n-sphere>`__).

    A distribution over this parameter space is represented using independent
    Power Spherical distributions for each vector. Constructing such a joint
    distribution requires :math:`count \\times (dimension + 1)` raw parameters
    as input.
    """

    count: int
    dimension: int
    target_entropy: List[float]

    def __init__(self, count: int, dimension: int):
        """
        Creates a :class:`Sphere` parameter space with ``count`` vectors in
        :math:`dimension`-dimensional space.

        :param count: Number of vectors
        :param dimension: Number of dimensions
        """
        if count < 1:
            raise ValueError("At least one value required")
        if dimension < 2:
            raise ValueError("At 2 dimensions required")

        self.count = count
        self.dimension = dimension
        self.target_entropy = [
            PowerSpherical(
                torch.tensor([1.0] + [0.0] * (dimension - 1)), torch.tensor(100.0)
            )
            .entropy()
            .item()
        ] * self.count

    def rand(self, batch_size: int) -> Tensor:
        """
        Returns a batch of uniformly-distributed samples from the parameter space.

        For :class:`Sphere`, this returns values of shape :math:`(batch, count, dimension)`.

        :param inputs: Batch size
        :return: Samples
        """
        return self.trans_forward(torch.rand(batch_size, self.trans_num_inputs))[0]

    @property
    def trans_num_inputs(self) -> int:
        """
        Returns the required number of points in :math:`[0, 1]` to be 
        transformed into a single point in the parameter space.

        For :class:`Sphere`, this is equal to :math:`count \\times dimension`.

        :return: Required number of input points
        """
        return self.count * self.dimension

    def trans_forward(self, inputs: Tensor) -> Tuple[Tensor, List[Any]]:
        """
        Transforms a set of points in :math:`[0, 1]` into a single point in the parameter space. 
        The transformation may also return extra objects which are passed to the planner.

        For :class:`Sphere`, this takes :math:`(batch, count \\times dimension)` samples and
        returns values of shape :math:`(batch, count, dimension)`.

        :param inputs: Input points
        :return: Parameter values and extra objects
        """
        assert len(inputs.shape) == 2
        assert inputs.shape[1] == self.trans_num_inputs
        y = inputs.reshape(-1, self.count, self.dimension)
        y = EPS + (1 - 2 * EPS) * y
        y = torch.special.ndtri(y)

        # When n is zero, the resultant vector is undefined.
        # We perturb the zero-ish elements until the norm is sufficient.
        n = np.linalg.norm(y, axis=-1, keepdims=True)
        while (n < 1e-8).any():
            y += (n < 1e-8) * (1e-4) * np.random.randn(*y.shape)
            n = np.linalg.norm(y, axis=-1, keepdims=True)

        return (y / n, [])

    @property
    def dist_num_inputs(self) -> int:
        """
        Returns the required number of raw parameters by a distribution
        over the parameter space.

        For :class:`Sphere`, this is equal to :math:`count \\times (dimension + 1)`.

        :return: Required number of raw parameters
        """
        return self.count * (self.dimension + 1)

    @property
    def dist_target_entropy(self) -> List[float]:
        """
        Returns the recommended target entropy for a distribution
        over the parameter space.

        For :class:`Sphere`, this returns :math:`count` values.

        :return: Target entropy
        """
        return self.target_entropy

    def create_sampler(self) -> Module:
        """
        Creates and returns a :class:`torch.nn.Module` which transforms
        raw distribution parameters into distribution samples and entropies
        in the parameter space. The sampler may also return extra objects
        which are passed to the planner.

        For :class:`Sphere`, the sampler takes raw distribution parameters of shape
        :math:`(batch, count \\times (dimension + 1))` and returns a tuple consisting of sampled
        values of shape :math:`(batch, count, dimension)` and entropy terms of shape
        :math:`(batch, count)`.

        :return: Sampler module
        """
        return SphereSampler(self.count, self.dimension)

    def create_embedding(self) -> Module:
        return torch.nn.Flatten(start_dim=1)
