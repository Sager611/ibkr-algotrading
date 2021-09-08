"""This expert maximizes the Sharpe ratio of a portfolio."""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from scipy.optimize import minimize

from modules.instruments import Portfolio, Stock
from modules.server import Server
from modules import utils

name = "Sharpe"
_LOGGER = logging.getLogger('ibkr-algotrading')


def init():
    pass


def get_returns(stocks, weights, start_date, end_date):
    ref = stocks[0].hist[start_date:end_date]
    # we retrieve the values for the stocks in the same dates
    # TODO: perhaps do this in parallel?
    stocks_value = np.stack(
        [utils.fill_like(stk.hist, ref)['close'].values
         for stk in stocks]
    ).T
    norm_vals = stocks_value / stocks_value[0, :]
    # normalized portfolio value by date
    norm_port_vals = (norm_vals * weights).sum(axis=1)
    rets = np.empty_like(norm_port_vals)
    rets[1:] = norm_port_vals[1:] / norm_port_vals[:-1] - 1

    # we ignore the 0 return
    rets = rets[1:]
    return rets


def predict(pf: Portfolio,
            date: pd.Timestamp,
            srv: Server,
            start_date: Optional[pd.Timestamp] = None,
            end_date: Optional[pd.Timestamp] = None) -> np.ndarray:
    """Given the input stocks, predict current wealth allocation needed to
    maximize wealth for :param:`date`.

    :param pf: portfolio containing stocks
    :param date: date at which we want to have maximum wealth
    :type date: pandas Timestamp
    :param srv: server context
    :type srv: :class:`modules.server.Server`
    :param start_date: start date for historical data.
        Defaults to `None`, which uses oldest available historical data for all stocks
    :type start_date: pandas Timestamp
    :param end_date: end date for historical data.
        Defaults to `None`, which uses the prediction date :param:`date` minus
        the stocks' bar
    :type end_date: pandas Timestamp
    :return: per-stock wealth distribution needed today so as to maximize
        total wealth for :param:`date`
    """
    stocks = pf.stocks

    if start_date is None:
        start_date = stocks[0].hist.index[0]
        for stk in stocks[1:]:
            d = stk.hist.index[0]
            if d > start_date:
                start_date = d
    if end_date is None:
        end_date = date - stocks[0].bar

    _LOGGER.info(f'Starting Sharpe ratio optimizer for dates: {start_date} -> {end_date}')

    # create constraint variable
    cons = ({'type': 'eq','fun': lambda w: np.sum(w) - 1.0})
    # create weight boundaries
    bounds = ((0, 1),) * len(stocks)
    # initial guess
    init_guess = [0.25] * len(stocks)
    # factor so sharpe ratio is the same no matter the bar.
    # we assume a 252 trading year
    K = np.sqrt(pd.Timedelta(days=252) / stocks[0].bar)

    def neg_sharpe(weights):
        returns = get_returns(stocks, weights, start_date, end_date)
        # we assume 0 risk-free rate
        sharpe = K * np.mean(returns) / np.std(returns)
        # log output
        w_str = '[' + ', '.join([f'{w:.2g}' for w in weights]) + ']'
        _LOGGER.info(f'sharpe: {sharpe : .2f} \n\t\tweights: {w_str}')
        return -sharpe

    opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    _LOGGER.info('Finished Sharpe ratio optimizer')
    return opt_results.x
