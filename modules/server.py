"""Server module to handle requests independently of the computing algorithm(s)."""

from functools import cached_property
import re
import time
import threading
import concurrent
import warnings
import logging
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd

from . import commands as cmd
from typing import Callable, Iterable, Optional, Union
from .instruments import Portfolio, Stock
from .utils import TracedThread

# max number of market history requests concurrently
NB_REQUEST_LIMIT = 5

_PERIOD_PATTERN = re.compile(r'(\d+)(min|h|d|w|m|y)')
_BAR_PATTERN = re.compile(r'(\d+)(min|h|d|w|m)')
_LOGGER = logging.getLogger('ibkr-algotrading')


class Server(object):
    """Server class.

    It also safeguards against possibly unwanted orders, due to bugs or unexpected
    behavior of the algorithm(s) being used.
    """

    _stock_cache: 'StockCache'
    _portfolios: list[Portfolio]
    _simulated: bool

    def __init__(self) -> None:
        self._stock_cache = StockCache(maxmem=100, maxstorage=200)
        self._portfolios = []
        self._simulated = False

    def start(self) -> None:
        # stock cache will auto-update the historical data with new info
        self._stock_cache.start()

    @cached_property
    def accountId(self) -> str:
        """IBKR Account ID to perform orders."""
        p_data = cmd.PortfoliosInfo()()
        if type(p_data) is not list:
            raise TypeError('Request for accountId failed.')

        if len(p_data) > 1:
            accIDs = ', '.join([str(dat["accountId"]) for dat in p_data])
            _LOGGER.warn(f'There are multiple account IDs! Using "{p_data[0]["accountId"]}". These are all of them: {accIDs}')

        return str(p_data[0]["accountId"])

    @property
    def balance(self, accountId: Optional[str] = None) -> tuple[float, str]:
        """Return for the given account the balance."""
        p_data = cmd.PortfoliosInfo()()

        if type(p_data) is not list:
            raise TypeError('Portfolio data is not a list.'
                            f' Data is: {p_data}')

        if accountId is None:
            accountId = p_data[0]["accountId"]
            if len(p_data) > 1:
                _LOGGER.warn(f'You have {len(p_data)} portfolios. Using the one with accountId: {accountId}')

        data = cmd.Balance()(accountId)
        currency = list(data.keys())[0]
        value = data[currency]["settledcash"]
        return value, currency

    def add_portfolio(self, pf: Portfolio) -> None:
        if hasattr(pf, 'server'):
            raise ValueError(f'The portfolio provided is already part of Server: {pf.server}')

        # sum of an empty numpy array is 0.0
        b = self.balance[0] - self.portfolios_wealth().sum()
        p_W = pf.wealth
        if b < p_W:
            raise ValueError('Cannot add portfolio because it allocates too much wealth. '
                             f'{p_W} > {b}')
        self._portfolios += [pf]
        pf.server = self

    def portfolios_wealth(self) -> np.ndarray:
        return np.array([pf.wealth for pf in self._portfolios])

    def get_portfolios(self) -> list[Portfolio]:
        """Return a copy of the server's portfolios.

        :return: numpy array of portfolios.
            The Portfolio objects are not copied.
        """
        return self._portfolios.copy()

    def simulated(self) -> '_SimulatedContext':
        """Return a simulated context to perform orders without actually spending money.

        Usage:

        .. code-block:: python

            with srv.simulated():
                ...
        """
        return _SimulatedContext(self)

    def __getitem__(self, args) -> Union[Stock, list[Stock]]:
        if args is None:
            raise ValueError('You must provide the symbol of a stock.')

        stock_names: Union[str, Iterable]
        period: str
        bar: str
        if type(args) is tuple:
            stock_names = args[0]
            period = '1m' if len(args) < 2 else args[1]
            bar = '1h' if len(args) < 3 else args[2]
        else:
            stock_names = args
            period = '1m'
            bar = '1h'

        if type(stock_names) is str:
            stock_names = [stock_names]
            was_stock_names_iterable = False
        elif hasattr(stock_names, '__iter__'):
            was_stock_names_iterable = True
        else:
            raise TypeError(f'`stock_names` is not iterable. It is of type: {type(stock_names)}')

        # check the requested period and bar make sense
        _validate_period_bar(period, bar)

        # Request stocks in parallel.
        # Order is preserved.
        # There's a limit of 5 concurrent requests.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # we only know that stock_names is iterable, so for convenience we make it a list
            names = [n for n in stock_names]
            stocks = []

            # max 5 concurrent market data requests according to IBKR
            for i in range(0, 1 + len(names) // NB_REQUEST_LIMIT):
                # async start requests
                futures = [
                    executor.submit(
                        self._stock_cache.__getitem__,
                        (name, period, bar)
                    )
                    for name in names[i * NB_REQUEST_LIMIT:(i+1) * NB_REQUEST_LIMIT]
                ]
                # join threads
                stocks += [f.result() for f in futures]

        # validate that stocks are correct
        for stock, name in zip(stocks, stock_names):
            _validate_stock_equality(name, period, bar, stock)

        if not was_stock_names_iterable:
            return stocks[0]
        return stocks

    def __del__(self) -> None:
        # deletion of the server means deletion of the stock cache's threads
        self._stock_cache.stop_threads()


def _validate_period_bar(period, bar):
    """Check inputs are what IBKR is expecting."""
    # period
    res = _PERIOD_PATTERN.findall(period)
    if len(res) == 0:
        raise ValueError(
                f'Period is in invalid format: {period} \n'
                'Expected format: {1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y'
        )

    p_val, p_unit = res[0]
    p_val = int(p_val)

    # bar
    res = _BAR_PATTERN.findall(bar)
    if len(res) == 0:
        raise ValueError(
                f'bar is in invalid format: {bar} \n'
                'Expected format: 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m'
        )

    b_val, b_unit = res[0]
    b_val = int(b_val)

    unit_to_val = {'min': 0, 'h': 1, 'd': 2, 'w': 3, 'm': 4, 'y': 5}
    p_unit_ = unit_to_val[p_unit]
    b_unit_ = unit_to_val[b_unit]
    if (
        b_unit_ > p_unit_
        or (b_unit_ == p_unit_ and b_val > p_val)
        or p_val == 0
    ):
        raise ValueError(f'Bar is larger than period: {bar} > {period}')

def _validate_stock_equality(stock_name, period, bar, target_stock):
    """Check if the requested stock is in fact equivalent to the stock found."""
    if target_stock.symbol != stock_name:
        raise ValueError(f'Requested stock (name "{stock_name}") does not match target stock (name "{target_stock.name}").')
    # TODO: check period and bar
    # if target_stock.period != period:
        # raise ValueError(f'Requested stock (period "{period}") does not match target stock (period "{target_stock.period}").')
    # if target_stock.bar != bar:
        # raise ValueError(f'Requested stock (bar "{bar}") does not match target stock (bar "{target_stock.bar}").')


class StockCache(object):
    """Automatically updates the historical of the cached stocks.

    TODO: implement :param:`maxstorage`, which indicates how many timeseries will
    be stored in the disk.

    :param maxmem: max number of entries in the cache
    :type maxmem: int
    :param _cache: contains as keys the stock's symbol and bar and values the
        stock (:class:`modules.Stock`) themselves.
    :type _cache: dict[tuple[str, str], Stock]
    """

    maxmem: int
    maxstorage: int
    _cache: dict[tuple[str, str], Stock]
    _cache_barriers: dict[tuple[str, str], threading.Lock]
    _main_barrier: threading.Lock
    _bar_threads: dict[str, TracedThread]
    _n_misses: int
    _callbacks: dict[str, list[Callable]]

    def __init__(self, maxmem: int = 128, maxstorage: int = 0) -> None:
        """Constructor."""
        self.maxmem = maxmem
        self.maxstorage = maxstorage
        self._cache = {}
        self._cache_barriers = {}
        self._main_barrier = threading.Lock()
        self._bar_threads = {}
        self._n_misses = 0
        self._callbacks = {}

    @property
    def n_misses(self) -> int:
        """Read-only number of cache misses."""
        return self._n_misses

    def clear(self) -> None:
        with self._main_barrier:
            self._cache.clear()
            self._cache_barriers.clear()
            self._bar_threads.clear()
            self._n_misses = 0

    def start(self) -> None:
        pass

    def add_callback(self, bar: str, func: Callable) -> None:
        with self._main_barrier:
            if bar not in self._callbacks:
                self._callbacks[bar] = []

            self._callbacks[bar] += [func]

    def clear_callbacks(self) -> None:
        with self._main_barrier:
            self._callbacks = {}

    def get_autoupdate_threads(self) -> dict[str, TracedThread]:
        """Return dictionnary of threads updating stocks' historical."""
        return self._bar_threads

    def _hist_update(self, bar: str, sleep_interval: float = 0.5) -> None:
        """Update all cached stocks with certain bar time-interval.

        This method is supposed to be executed in a thread using
        :class:`modules.utils.TracedThread`.
        """
        # adhere to the bar's time grid
        dt = _ibkr_to_timedelta(bar)
        dt = dt.total_seconds()
        today = datetime.today().timestamp()
        offset = dt - (today - dt * int(today / dt))

        _LOGGER.info(f'Started StockCache thread for bar "{bar}". \n\t\t'
                     f'Next data update will be in {offset : .2f}s.')
        # sleep in intervals to allow for kill interrupts
        for _ in range(int(offset // sleep_interval)):
            time.sleep(sleep_interval)
        time.sleep(offset % sleep_interval)

        def _stk_thread(stk):
            # this method is called from a thread that locks
            # the cache, so access to stk should not collide
            # with other threads.
            # add just enough data to fill prices until today
            today = datetime.today().timestamp()
            secs = (today - stk.hist.index[-1].timestamp())
            period = _seconds_to_ibkr(secs)

            _validate_period_bar(period, bar)
            _update_stock(stk, period, bar)

            _LOGGER.info(f'Updated: {str(stk)} - bar: {bar} - added period: {period}')

        def _start_callbacks(cbs):
            for func in cbs:
                func(stocks)

        # now that we are in the same time grid as the bar,
        # loop re-requesting the stocks' historical info
        while True:
            cbs = []
            # start critical section
            with self._main_barrier:
                stocks = []
                # retrieve all stocks with this bar
                for key, stk in self._cache.items():
                    if key[1] == bar:
                        stocks += [stk]

                # Request stocks in parallel.
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # max 5 concurrent market data requests according to IBKR
                    for i in range(0, 1 + len(stocks) // NB_REQUEST_LIMIT):
                        # async start requests
                        futures = [
                            executor.submit(_stk_thread, stk)
                            for stk in stocks[i * NB_REQUEST_LIMIT:(i+1) * NB_REQUEST_LIMIT]
                        ]
                        # join threads
                        for f in futures:
                            f.result()

                # callbacks
                if bar in self._callbacks:
                    cbs = self._callbacks[bar].copy()

            # end critical section

            # execute callbacks in another thread with timeout of 60s
            cbs_thread = TracedThread(target=_start_callbacks, args=(cbs,), timeout=60.0)
            cbs_thread.start()

            # sleep until next request
            today = datetime.today().timestamp()
            offset = dt - (today - dt * int(today / dt))
            # prevent rapid-fire requests
            if offset < 0.1:
                # make sure we go past the time target this time
                time.sleep(0.2)
                # .. and recalculate when next update is required
                today = datetime.today().timestamp()
                offset = dt - (today - dt * int(today / dt))
            # sleep in intervals to allow for kill interrupts
            for _ in range(int(offset // sleep_interval)):
                time.sleep(sleep_interval)
            time.sleep(offset % sleep_interval)

    def __getitem__(self, args) -> Stock:
        name: str = args[0]
        period: str = args[1]
        bar: str = args[2]

        key = (name, bar)
        # we are in a critical section
        self._main_barrier.acquire()
        try:
            if key in self._cache:
                with self._cache_barriers[key]:
                    stock = self._cache[key]
                # if stored stock's period is shorter, we need to update it
                if pd.Timestamp.today() - stock.hist.index[0] >= _ibkr_to_timedelta(period):
                    # end of critical section
                    self._main_barrier.release()
                    return stock
                else:
                    self._cache_barriers[key].acquire()
            # this condition should not be possibly true! (since we init the _cache to None)
            elif key in self._cache_barriers:
                warnings.warn(f'A race condition happened when requesting key "{key}"!')
                with self._cache_barriers[key]:
                    stock = self._cache[key]
                # if stored stock's period is shorter, we need to update it
                if pd.Timestamp.today() - stock.hist.index[0] >= _ibkr_to_timedelta(period):
                    # end of critical section
                    self._main_barrier.release()
                    return stock
            else:
                # increment miss count
                self._n_misses += 1

                # Evict according to LRU policy.
                # This is possible thanks to the order-preserving
                # python dicts.
                keys = list(self._cache.keys())
                if len(keys) >= self.maxmem:
                    self._cache.pop(keys[0])

                # create lock for cache entry
                self._cache_barriers[key] = threading.Lock()
                # temporarely add cache entry
                self._cache[key] = None
                # immediately lock it
                self._cache_barriers[key].acquire()

                # create hist auto-update if it does not exist
                if bar not in self._bar_threads:
                    self._bar_threads[bar] = TracedThread(target=self._hist_update, args=(bar,))
                    self._bar_threads[bar].start()
        except BaseException as e:
            # it's important to release the lock, even with an exception ;)
            self._main_barrier.release()
            raise e

        # end of critical section
        self._main_barrier.release()

        # request new stock
        stock = _request_stock(name, period, bar)
        self._cache[key] = stock

        # unlock this stock's mutex
        self._cache_barriers[key].release()

        return stock

    def stop_threads(self) -> None:
        # make sure running threads are stopped
        try:
            _LOGGER.info(f'Trying to stop {len(self._bar_threads)} threads..')
            for t in self._bar_threads.values():
                t.kill()
            for t in self._bar_threads.values():
                t.join()
            _LOGGER.info('Successfully stopped all StockCache threads.')
        except BaseException as e:
            warnings.warn('Could not stop StockCache\'s threads. \n\t\t'
                          f'Raised Exception: {e}')


def _request_stock(stock_name: str, period: str, bar: str) -> Stock:
    """Return stock specified by the arguments and stores it following a LRU cache."""
    info_data = cmd.StockInfo()(stock_name)
    hist_data = cmd.MarketDataHistory()(info_data[0]["conid"], period=period, bar=bar)
    return Stock(info_data, hist_data)


def _ibkr_to_timedelta(period: str) -> pd.Timedelta:
    """From IBKR's market data time interval format to pandas Timedelta."""
    res = _PERIOD_PATTERN.findall(period)
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


def _update_stock(stk: Stock, period: str, bar: str):
    hist_data = cmd.MarketDataHistory()(stk.conid, period=period, bar=bar)
    stk.insert(hist_data)


def _seconds_to_ibkr(s: float) -> str:
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


class _SimulatedContext(object):
    """Class used by :class:`modules.Server` to simulated orders."""
    _server: Server

    def __init__(self, srv: Server) -> None:
        self._server = srv

    def __enter__(self) -> None:
        _LOGGER.warn('ENTERING SIMULATED CONTEXT. ALL FOLLOWING TRANSACTIONS WILL BE SIMULATED.')
        self._server._simulated = True
        for pf in self._server._portfolios:
            pf._simulated = True

    def __exit__(self, type, value, tb) -> None:
        _LOGGER.warn('LEAVING SIMULATED CONTEXT. ALL FOLLOWING TRANSACTIONS WILL BE REAL.')
        self._server._simulated = False
        for pf in self._server._portfolios:
            pf._simulated = False
