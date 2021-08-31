"""Utility methods and classes for global usage."""

import re
import sys
import time
import logging
import threading
from math import ceil

import pandas as pd
import numpy as np

# IBKR's fraction of wealth that is taken as commission
IBKR_COMMISSION_FRACTION = 0.01
PERIOD_PATTERN = re.compile(r'(\d+)(min|h|d|w|m|y)')
BAR_PATTERN = re.compile(r'(\d+)(min|h|d|w|m)')

_LOGGER = logging.getLogger('ibkr-algotrading')


class TracedThread(threading.Thread):
    """This class allows threads which can be killed.

    Note that if the thread calls `time.sleep()`, it won't receive signals and
    thus cannot be stopped in the duration it is asleep.

    :arg timeout:
    :type timeout: float
    """
    def __init__(self, *args, timeout: float = None, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.killed = False
        self._start_t = time.time()
        self.timeout = timeout

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.timeout and time.time() - self._start_t >= self.timeout:
            self.killed = True
            _LOGGER.warn(f'Killed thread for exceeding its timeout of {self.timeout:.2f}s.')
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


def ibkr_to_timedelta(period: str) -> pd.Timedelta:
    """From IBKR's market data time interval format to pandas Timedelta."""
    res = PERIOD_PATTERN.findall(period)
    if len(res) == 0:
        raise ValueError(
                f'Period is in invalid format: {period} \n'
                'Expected format: {1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y'
        )

    b_val, b_unit = res[0]
    if b_unit == 'm':
        return pd.Timedelta(value=int(b_val) * 30.44, unit='d')
    if b_unit == 'y':
        return pd.Timedelta(value=int(b_val) * 365.2425, unit='d')
    return pd.Timedelta(value=int(b_val), unit=b_unit)


def seconds_to_ibkr(s: float) -> str:
    """From seconds to IBKR's market data time interval format."""
    if s <= 0:
        raise ValueError(f'Time cannot be 0 or negative (it is {s}s)')

    min = ceil(s / 60.0)
    if min <= 30:
        return f'{min}min'

    h = ceil(s / 3600.0)
    if h <= 8:
        return f'{h}h'

    d = ceil(s / 86400.0)
    if d <= 1000:
        return f'{d}d'

    w = ceil(s / 604800.0)
    if w <= 792:
        return f'{w}w'

    m = ceil(s / 16934400.0)
    if m <= 182:
        return f'{m}m'

    y = ceil(s / 220898664.0)
    if y <= 15:
        return f'{y}y'

    raise ValueError(f'Time way too big: {s}s')


def get_transaction_commission(transact: np.ndarray, diff: np.ndarray, stocks: np.ndarray, srv) -> float:
    """Return the commission taken by IBKR after buy/sell transactions.

    :param transact: buy (>0)/ sell (<0) transactions that will take place in units of currency
    :param diff: shares being bought (>0)/sold (<0) for each stock
    :param stocks: stocks being bought/sold
    :param srv: Server context
    :return: commission.
    :rtype: float
    """
    # TODO: commissions are probably more complex than this
    comm = np.abs(transact).sum() * IBKR_COMMISSION_FRACTION
    return comm

def fill_like(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Return copy of df1 with added rows so as to match the indices in df2.

    Inputs are assumed to be historical data.
    """
    ret = df1.iloc[0:0, :].copy()
    for i in df2.index:
        j = df1.index.get_loc(i, method='ffill')
        ret.loc[i] = df1.iloc[j]
    return ret
