from typing import Any, Generic, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from ..domain import Domain
from ..generator import Generator
from ..parameter_space import ParameterSpace

EPS = 1e-6

Problem = TypeVar("Problem")


def create_output_layer(spaces: List[ParameterSpace]) -> nn.Module:
    return nn.LazyLinear(sum(s.dist_num_inputs for s in spaces))


def outputs_to_parameters(
    outputs: torch.Tensor, spaces: List[ParameterSpace], samplers: nn.ModuleList
) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], List[List[Any]]]:
    assert len(outputs.shape) == 2, "Invalid outputs"
    assert outputs.shape[1] == sum(s.dist_num_inputs for s in spaces), "Invalid outputs"

    parameters = []
    entropies: List[torch.Tensor] = []
    extras: List[List[Any]] = []
    offset = 0
    for (pspace, sampler) in zip(spaces, samplers):
        (p, e, o) = sampler(outputs[:, offset : offset + pspace.dist_num_inputs])
        parameters.append(p)
        if e is not None:
            entropies.append(e)
        if len(o) > 0:
            extras.append(o)
        offset += pspace.dist_num_inputs
    return (
        parameters,
        torch.concat(entropies, dim=-1) if len(entropies) > 0 else None,
        extras,
    )


class FCResNetGenerator(Generic[Problem], Generator[Problem]):
    """
    :class:`FCResNetGenerator` represents a generator :math:`G_\\theta(c)` implemented using a
    residual network of fully connected hidden layers with skip connections.
    """

    parameter_spaces: List[ParameterSpace]
    latent_size: int
    block_size: int
    num_blocks: int

    def __init__(
        self,
        domain: Domain[Problem],
        latent_size: int = 512,
        block_size: int = 2,
        num_blocks: int = 4,
        batch_norm: bool = False,
    ):
        """
        Constructs a :class:`FCResNetGenerator` using embedding and parameter space
        information from a given :class:`opof.Domain`.

        :param domain: Domain
        :param latent_size: Size of hidden layers
        :param block_size: Number of hidden layers in each residual block
        :param num_blocks: Number of residual blocks
        """
        super(FCResNetGenerator, self).__init__()
        self.parameter_spaces = domain.composite_parameter_space()
        self.latent_size = latent_size
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.problem_embedding = domain.create_problem_embedding()
        self.fc_pre = nn.Sequential(nn.LazyLinear(latent_size), nn.GELU())
        self.fc = nn.ModuleList(
            [
                nn.Linear(latent_size, latent_size)
                for _ in range(block_size * num_blocks)
            ]
        )
        if batch_norm:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(latent_size) for _ in range(block_size * num_blocks)]
            )
        else:
            self.bn = [lambda x: x for _ in range(block_size * num_blocks)]

        self.output_layer = nn.Linear(
            latent_size, sum(pspace.dist_num_inputs for pspace in self.parameter_spaces)
        )
        self.samplers = nn.ModuleList(
            [pspace.create_sampler() for pspace in self.parameter_spaces]
        )

    def forward(
        self, problem: List[Problem]
    ) -> Tuple[List[Tensor], Optional[Tensor], List[List[Any]]]:
        # Embed problem.
        x = self.problem_embedding(problem)
        x = self.fc_pre(x)

        # Res block.
        x_skip = x
        for i in range(self.num_blocks):
            for j in range(self.block_size):
                index = i * self.block_size + j
                if j < self.block_size - 1:
                    x = F.gelu(self.bn[index](self.fc[index](x)))
                else:
                    x_skip = x = F.gelu(self.bn[index](self.fc[index](x)) + x_skip)

        # Output layer.
        x = self.output_layer(x)

        # Parameters.
        return outputs_to_parameters(x, self.parameter_spaces, self.samplers)


class FCResNetCritic(nn.Module, Generic[Problem]):
    """
    :class:`FCResNetCritic` represents a differentiable surrogate model for
    the planning objective :math:`\\boldsymbol{f}(x; c)`. It is implemented
    using a residual network of fully connected hidden layers with skip connections.
    The output is modelled with a Normal distribution.
    """

    parameter_spaces: List[ParameterSpace]
    latent_size: int
    block_size: int
    num_blocks: int

    def __init__(
        self,
        domain: Domain[Problem],
        latent_size: int = 512,
        block_size: int = 2,
        num_blocks: int = 4,
        batch_norm: bool = False,
    ):
        """
        Constructs a :class:`FCResNetCritic` using embedding and parameter space
        information from a given :class:`opof.Domain`.

        :param domain: Domain
        :param latent_size: Size of hidden layers
        :param block_size: Number of hidden layers in each residual block
        :param num_blocks: Number of residual blocks
        """
        super(FCResNetCritic, self).__init__()
        self.parameter_spaces = domain.composite_parameter_space()
        self.latent_size = latent_size
        self.block_size = block_size
        self.num_blocks = num_blocks

        self.problem_embedding = domain.create_problem_embedding()
        self.parameters_embedding = nn.ModuleList(
            [pspace.create_embedding() for pspace in self.parameter_spaces]
        )
        self.fc_pre = nn.Sequential(nn.LazyLinear(latent_size), nn.GELU())
        self.fc = nn.ModuleList(
            [
                nn.Linear(latent_size, latent_size)
                for _ in range(block_size * num_blocks)
            ]
        )
        if batch_norm:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(latent_size) for _ in range(block_size * num_blocks)]
            )
        else:
            self.bn = [lambda x: x for _ in range(block_size * num_blocks)]
        self.output_layer = nn.Linear(latent_size, 2)

    def forward(
        self, problem: List[Problem], parameters: List[torch.Tensor]
    ) -> torch.distributions.Normal:
        """
        Transforms a batch of problem instances :math:`c \\in \\mathcal{C}` and a
        batch of corresponding planning parameters :math:`x \\in \\mathcal{X}`
        into a batch of Normal distributions.

        :return: Batch of Normal distributions
        """
        # Embed problem.
        x_problem = self.problem_embedding(problem)
        # Embed parameters.
        x_parameters = [
            pse(p) for (p, pse) in zip(parameters, self.parameters_embedding)
        ]

        # Concat problem and parameter embeddings.
        x = torch.concat([x_problem] + x_parameters, dim=-1)
        x = self.fc_pre(x)

        # Res block.
        x_skip = x
        for i in range(self.num_blocks):
            for j in range(self.block_size):
                index = i * self.block_size + j
                if j < self.block_size - 1:
                    x = F.gelu(self.bn[index](self.fc[index](x)))
                else:
                    x_skip = x = F.gelu(self.bn[index](self.fc[index](x)) + x_skip)

        # Output layer.
        x = self.output_layer(x)
        x = x.reshape(-1, 2)
        loc = x[..., 0]
        scale = EPS + F.softplus(x[..., 1])
        return Normal(loc, scale)
