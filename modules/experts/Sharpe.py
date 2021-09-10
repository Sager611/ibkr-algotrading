"""This expert maximizes the Sharpe ratio of a portfolio."""

import time
import logging
from typing import Optional
from concurrent.futures.thread import ThreadPoolExecutor

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


def get_returns(weights, stocks_value):
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
    init_guess = [1 / len(stocks)] * len(stocks)
    # factor so sharpe ratio is the same no matter the bar.
    # we assume a 252 trading year
    K = np.sqrt(pd.Timedelta(days=252) / stocks[0].bar)

    t1 = time.perf_counter()
    ref = stocks[0].hist[start_date:end_date]
    # we retrieve the values for the stocks in the same dates
    stocks_value = [ref['close'].values]
    with ThreadPoolExecutor() as executor:
        # async start requests
        futures = [
            executor.submit(
                lambda s: utils.fill_like(s.hist, ref)['close'].values, stk)
            for stk in stocks[1:]
        ]
        for stk, f in zip(stocks, futures):
            stocks_value += [f.result()]
            print(f'Done: {stk}')
    stocks_value = np.stack(stocks_value).T
    t2 = time.perf_counter()
    _LOGGER.info(f'Retrieving stock values took {t2-t1:.4g}s')

    def neg_sharpe(weights):
        returns = get_returns(weights, stocks_value)
        # we assume 0 risk-free rate
        sharpe = K * np.mean(returns) / np.std(returns)
        # log output
        _LOGGER.info(f'sharpe: {sharpe : .2f}')
        return -sharpe

    opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    # log output
    w_str = '[' + ', '.join([f'{w:.2g}' for w in opt_results.x]) + ']'
    _LOGGER.info(f'Finished Sharpe ratio optimizer. \n\t\tOptimal weights: {w_str}')
    return opt_results.x
