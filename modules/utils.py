"""Utility methods and classes for global usage."""

import re
import sys
import time
import trace
import logging
import threading
from math import ceil

import pandas as pd

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


def get_adjusted_wealth(quantity: float, n_orders: int = 1) -> float:
    """Return the amount of wealth we can use in a single order so as to match :param:`quantity`.

    If we have 100. USD and want to use it all to buy and/or sell some stocks, we can't simply
    perform the transaction as we will spend 100. * (1. + COMMISSION) USD.
    This function returns the amount X required to spend X * (1 + COMMISSION) = 100. USD in
    the transactions.

    :param quantity: total wealth we have at our disposition to spend
    :param n_orders: how many times we'll spend all of our wealth in transactions.
        Defaults to 1.
    :return: actual amount of wealth we should use in the transactions in order to, in the end,
        spend the provided input :param:`quantity`.
    """
    return quantity / (1. + IBKR_COMMISSION_FRACTION) ** n_orders
