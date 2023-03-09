from abc import abstractmethod
from typing import Optional

from .domain import Domain


class Algorithm:
    """
    :class:`Algorithm` is the abstract base class representing an algorithm to learn 
    a generator for a given domain. 

    .. warning::
        Classes that derive from :class:`Algorithm` **must** have a constructor which 
        accepts: (a) a first parameter of type :class:`Domain`, and (b) two named parameters 
        ``eval_folder`` and ``save_folder`` of type :class:`Optional[str]`. Failure to do so
        may cause your code to not run as expected.
    """
    domain: Domain
    eval_folder: Optional[str]
    save_folder: Optional[str]

    def __init__(
        self, domain: Domain, eval_folder: Optional[str], save_folder: Optional[str]
    ):
        """
        Constructs an instance of the algorithm for a given domain.

        :param domain: Domain
        :param eval_folder: Optional path to write evaluation logs across training
        :param save_folder: Optional path to save models across training
        """
        self.domain = domain
        self.eval_folder = eval_folder
        self.save_folder = save_folder

    @abstractmethod
    def __call__(self):
        """
        Runs the algorithm.
        """
