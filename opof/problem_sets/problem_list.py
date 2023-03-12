import random
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

from ..problem_set import ProblemSet

Problem = TypeVar("Problem")


class ProblemList(Generic[Problem], ProblemSet[Problem]):
    """
    :class:`ProblemList` represents a distribution over a fixed list of problem instances formed
    by repeatedly shuffling the list and iterating through it.
    """

    data: List[Problem]
    index: int

    def __init__(self, data: List[Problem], shuffle=True):
        """
        Creates a :class:`ProblemList` using a fixed list of problem instances.

        :param data: List of problem instances
        :param shuffle: Whether to shuffle the list
        """
        self.data = data[:]  # Copy list.
        self.shuffle = shuffle
        self.index = 0
        if self.shuffle:
            random.shuffle(self.data)

    def __call__(
        self,
        previous: Optional[Tuple[Problem, List[np.ndarray], Dict[str, float]]] = None,
    ) -> Problem:
        """
        Returns a problem instance from the problem set.

        For :class:`ProblemList`, this repeatedly shuffles and iterates through the fixed list of problem instances.

        :param previous: Ignored
        :return: Problem instance
        """
        result = self.data[self.index]
        self.index += 1
        if self.index >= len(self.data):
            if self.shuffle:
                random.shuffle(self.data)
            self.index = 0
        return result
