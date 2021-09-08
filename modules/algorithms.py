"""Contains meta-algorithms which exploit bagging of other lower level algorithms ('experts')."""

from abc import abstractmethod

import numpy as np

from .server import Server


class BaseAlgorithm(object):
    """Base Algorithm."""
    experts: list

    def __init__(self, srv: Server) -> None:
        self.server = srv
        self.experts = []

    def start(self) -> None:
        raise NotImplementedError()

    def add(self, expert) -> None:
        """Insert expert algorithm."""
        _validate_expert(expert)
        self.experts += [expert]

    @abstractmethod
    def output(self, expert_outputs: np.ndarray) -> None:
        raise ValueError('This method is abstract.')


def _validate_expert(exp):
    if not hasattr(exp, 'name'):
        raise RuntimeError('Expert is missing "name"')
    if not hasattr(exp, 'init'):
        raise RuntimeError(f'Expert "{exp.name}" is missing "init" method')
    if not callable(exp.init):
        raise RuntimeError(f'Expert "{exp.name}" has non-callable "init" attribute')


class Hedge(BaseAlgorithm):
    """Hedge algorithm."""
