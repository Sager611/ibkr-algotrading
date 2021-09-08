"""Contains meta-algorithms which exploit bagging of other lower level algorithms ('experts')."""

from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from modules.server import Server
from modules.instruments import Portfolio


class BaseAlgorithm(object):
    """Base Algorithm class."""
    server: Server
    experts: list
    portfolio: Portfolio
    bar: pd.Timedelta

    def __init__(self, srv: Server, pf: Portfolio, bar: pd.Timedelta) -> None:
        self.server = srv
        self.experts = []
        self.portfolio = pf
        # how far in the future to predict
        self.bar = bar

        # step will be called every time stocks are updated
        srv.add_callback(self.step)

    @abstractmethod
    def start(self) -> None:
        raise RuntimeError('This method is abstract.')

    def add(self, expert) -> None:
        """Insert expert algorithm."""
        _validate_expert(expert)
        self.experts += [expert]

    def step(self) -> None:
        """Perform a step in the algorithm.

        This method is called every time the stock's historical data is updated,
        and should be called in the :meth:`start` method.
        """
        date = pd.Timestamp.today() + self.bar
        expert_outputs = np.array([
            exp.predict(
                pf=self.portfolio,
                date=date,
                start_date=None,
                end_date=None,
                srv=self.server)
            for exp in self.experts
        ])
        weights = self.output(expert_outputs, date)
        # if None, do not perform an order
        if weights is None:
            return
        else:
            self.portfolio.order(weights)

    @abstractmethod
    def output(self, expert_outputs: np.ndarray, date: pd.Timestamp) -> Optional[np.ndarray]:
        raise RuntimeError('This method is abstract.')


def _validate_expert(exp):
    if not hasattr(exp, 'name'):
        raise RuntimeError('Expert is missing "name"')
    if not hasattr(exp, 'init'):
        raise RuntimeError(f'Expert "{exp.name}" is missing "init" method')
    if not callable(exp.init):
        raise RuntimeError(f'Expert "{exp.name}" has non-callable "init" attribute')
    if not hasattr(exp, 'predict'):
        raise RuntimeError(f'Expert "{exp.name}" is missing "predict" method')
    if not callable(exp.predict):
        raise RuntimeError(f'Expert "{exp.name}" has non-callable "predict" attribute')


class Hedge(BaseAlgorithm):
    """Hedge algorithm."""
    # TODO: implement Hedge algorithm

    def start(self) -> None:
        for exp in self.experts:
            exp.init()
        self.step()

    def output(self, expert_outputs: np.ndarray, date: pd.Timestamp) -> Optional[np.ndarray]:
        return expert_outputs[0]
