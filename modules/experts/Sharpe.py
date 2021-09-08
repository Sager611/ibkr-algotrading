"""This expert maximizes the Sharpe ratio of a portfolio."""

import pandas as pd
import numpy as np

from modules.server import Server

name = "Sharpe"

def init():
    pass


def predict(stock_values: pd.DataFrame, date: pd.Timestamp, srv: Server) -> np.ndarray:
    """Given the input stock values, predict wealth allocation for :param:`date`."""
    pass
