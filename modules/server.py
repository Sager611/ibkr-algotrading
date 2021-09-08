"""Server module to handle requests independently of the computing algorithm(s)."""

import time
import threading
import warnings
import logging
from pathlib import Path
from functools import cached_property
from datetime import datetime
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import commands as cmd
from . import utils
from typing import Callable, Iterable, Optional, Union
from .instruments import Portfolio, Stock
from .utils import TracedThread

# max number of market history requests concurrently
NB_REQUEST_LIMIT = 5

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
        self._stock_cache = StockCache(maxmem=100, maxstorage=500)
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

        # if we are in a simulated context
        if self._simulated:
            _LOGGER.warn('Showing real balance, not simulated one.')

        data = cmd.Balance()(accountId)
        currency = list(data.keys())[0]
        value = data[currency]["settledcash"]
        return value, currency

    def risk_free_rot(self,
                      date: Optional[pd.Timestamp] = None,
                      dt: pd.Timedelta = pd.Timedelta(days=1),
                      period: str = '1y',
                      bar: str = '1d'
                      ) -> tuple[float, Optional[pd.Timestamp]]:
        """Risk free rate of return.

        Reference:
        Ford, G. S. (2019). Estimating Betas in Practice: Alternatives that Matter and Those that Do Not.
        https://phoenix-center.org/perspectives/Perspective19-01Final.pdf
        THIS FUNCTION MAY BE TOTALLY WRONG, I'M NOT RESPONSIBLE FOR ANY DAMAGES
        """
        # IRX: 13 Week Treasury Bill
        irx = self['IRX', period, bar]
        if date is not None:
            # first, check if we have to update the historical of the stock
            if date < irx.hist.index[0]:
                today = datetime.today().timestamp()
                bar_secs = irx.bar.total_seconds()
                # we take the period to the given date + 13 weeks
                weeks13_secs = pd.Timedelta(days=13*7).total_seconds()
                period_secs = bar_secs + (today - date.timestamp()) + weeks13_secs

                period = utils.seconds_to_ibkr(period_secs)

                # by requesting the stock from the server,
                # we are also updating its historical
                self[irx.symbol, period, bar]
        # we are calculating the yield at t-1
        val_t_1, rf_date = irx.value(date)
        # for some reason IBKR's IRX are in bps, or tenths of a percent
        y_t_1 = val_t_1 / 10.0 / 100.0

        q = utils.ibkr_to_timedelta('13w') / dt

        rf = (1 / (1 - y_t_1 * 0.25)) ** (1/q) - 1
        if date is None:
            rf_date = pd.Timestamp.today()
        return rf, rf_date

    def add_portfolio(self, pf: Portfolio) -> None:
        if hasattr(pf, 'server'):
            raise ValueError(f'The portfolio provided is already part of Server: {pf.server}')

        # sum of an empty numpy array is 0.0
        b = self.balance[0] - self.portfolios_wealth().sum()
        p_W = pf.wealth()
        if b < p_W:
            raise ValueError('Cannot add portfolio because it allocates too much wealth. '
                             f'{p_W} > {b}')
        self._portfolios += [pf]
        pf.server = self

    def portfolios_wealth(self) -> np.ndarray:
        # TODO: may be smart to do this in parallel
        return np.array([pf.wealth() for pf in self._portfolios])

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

    def copy_state(self) -> '_ServerState':
        return _ServerState(self)

    def set_state(self, state: '_ServerState') -> None:
        self._stock_cache.set(state._stock_cache)
        # we pray that the order of the portfolios was kept as they were being added
        new_portfolios = []
        for i, pf in enumerate(state._portfolios):
            spf = self._portfolios[i]
            spf.set(pf)
            new_portfolios += [spf]
        self._portfolios = new_portfolios
        self._simulated = state._simulated

    def __getitem__(self, args) -> Union[Stock, list[Stock]]:
        if args is None:
            raise ValueError('You must provide the symbol of a stock.')

        stock_names: Union[str, Iterable]
        period: str
        bar: str
        extra_args: dict
        if type(args) is tuple:
            stock_names = args[0]
            period = '1m' if len(args) < 2 else args[1]
            bar = '1h' if len(args) < 3 else args[2]
            extra_args = {} if len(args) < 4 else args[3]
        else:
            stock_names = args
            period = '1m'
            bar = '1h'
            extra_args = {}

        if "ignore_errors" in extra_args:
            ignore_errors = extra_args["ignore_errors"]
        else:
            ignore_errors = False

        if type(stock_names) is str:
            stock_names = [stock_names]
            was_stock_names_iterable = False
        elif hasattr(stock_names, '__iter__'):
            was_stock_names_iterable = True
        else:
            raise TypeError(f'`stock_names` is not iterable. It is of type: {type(stock_names)}')

        # check the requested period and bar make sense
        _validate_period_bar(period, bar)

        def _stk_getter(args):
            try:
                stk = self._stock_cache[args]
                return stk
            except Exception as e:
                if ignore_errors:
                    _LOGGER.error(f'Exception raised for stock "{args[0]}": \n\t\t{e}')
                    return None
                else:
                    raise e

        # Request stocks in parallel.
        # Order is preserved.
        # There's a limit of 5 concurrent requests.
        with ThreadPoolExecutor(max_workers=NB_REQUEST_LIMIT) as executor:
            # we only know that stock_names is iterable, so for convenience we make it a list
            names = [n for n in stock_names]

            # async start requests
            futures = [
                executor.submit(
                    _stk_getter,
                    (name, period, bar)
                )
                for name in names
            ]
            # join threads
            stocks = [f.result() for f in futures]

            # remove None values arising from handled exceptions
            stocks_, stock_names_ = [], []
            for stk, name in zip(stocks, stock_names):
                if stk is not None:
                    stocks_ += [stk]
                    stock_names_ += [name]
            stocks = stocks_
            stock_names = stock_names_

        # validate that stocks are correct
        for stk, name in zip(stocks, stock_names):
            _validate_stock_equality(name, period, bar, stk)

        if not was_stock_names_iterable:
            return stocks[0]
        return stocks

    def __del__(self) -> None:
        # deletion of the server means deletion of the stock cache's threads
        self._stock_cache.stop_threads()


def _validate_period_bar(period, bar):
    """Check inputs are what IBKR is expecting."""
    # period
    res = utils.PERIOD_PATTERN.findall(period)
    if len(res) == 0:
        raise ValueError(
                f'Period is in invalid format: {period} \n'
                'Expected format: {1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y'
        )

    # bar
    res = utils.BAR_PATTERN.findall(bar)
    if len(res) == 0:
        raise ValueError(
                f'bar is in invalid format: {bar} \n'
                'Expected format: 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m'
        )

    if (utils.ibkr_to_timedelta(bar) > utils.ibkr_to_timedelta(period)):
        raise ValueError(f'Bar is larger than period: {bar} > {period}')

def _validate_stock_equality(stock_name, period, bar, target_stock):
    """Check if the requested stock is in fact equivalent to the stock found."""
    if target_stock.symbol != stock_name:
        raise ValueError(f'Requested stock (name "{stock_name}") does not match target stock (name "{target_stock.symbol}").')
    # TODO: check period and bar
    # if target_stock.period != period:
        # raise ValueError(f'Requested stock (period "{period}") does not match target stock (period "{target_stock.period}").')
    # if target_stock.bar != bar:
        # raise ValueError(f'Requested stock (bar "{bar}") does not match target stock (bar "{target_stock.bar}").')


class StockCache(object):
    """Automatically updates the historical of the cached stocks.

    You can specify :param:`maxstorage`, which indicates how many timeseries will
    be stored in the disk. By default, a directory called `.cache/` will be created
    in the project's root directory.

    :param maxmem: max number of entries in the cache
    :type maxmem: int
    :param maxstorage: max number of entries saved in disk.
        If set to 0, then no entries will be saved in disk.
    :type maxstorage: int
    :param _cache: contains as keys the stock's symbol and bar and values the
        stock (:class:`modules.Stock`) themselves.
    :type _cache: dict[tuple[str, str], Stock]
    """

    maxmem: int
    maxstorage: int
    save_path: Path
    _cache: dict[tuple[str, str], Stock]
    _cache_barriers: dict[tuple[str, str], threading.Lock]
    _main_barrier: threading.Lock
    _bar_threads: dict[str, TracedThread]
    _n_misses: int
    _n_storage_misses: int
    _callbacks: dict[str, list[Callable]]

    def __init__(self, maxmem: int = 128, maxstorage: int = 0) -> None:
        """Constructor."""
        self.maxmem = maxmem
        self.maxstorage = maxstorage
        # basically, "../.cache"
        self.save_path = Path(__file__).parent.parent.joinpath('.cache')
        self._cache = {}
        self._cache_barriers = {}
        self._main_barrier = threading.Lock()
        self._bar_threads = {}
        self._n_misses = 0
        self._n_storage_misses = 0
        self._callbacks = {}

    @property
    def n_misses(self) -> int:
        """Read-only number of cache misses."""
        return self._n_misses

    def copy(self) -> 'StockCache':
        """Returned stock cache needs to be started again to start threads."""
        ins = StockCache(self.maxmem, self.maxstorage)
        ins._cache = {
            k: stk.copy() for k, stk in self._cache.items()
        }
        # locks in the copy will be unlocked
        ins._cache_barriers = {
            k: threading.Lock() for k in self._cache_barriers.keys()
        }
        ins._main_barrier = threading.Lock()
        # threads have to be started after copy
        ins._bar_threads = {
            bar:  TracedThread(target=self._hist_update, args=(bar,))
            for bar in self._bar_threads.keys()
        }
        ins._n_misses = self._n_misses
        ins._n_storage_misses = self._n_storage_misses
        ins._callbacks = self._callbacks.copy()
        return ins

    def set(self, sc: 'StockCache') -> None:
        self._cache = sc._cache
        self._cache_barriers = sc._cache_barriers
        self._main_barrier = sc._main_barrier
        self._bar_threads = sc._bar_threads
        self._n_misses = sc._n_misses
        self._n_storage_misses = sc._n_storage_misses
        self._callbacks = sc._callbacks

    def clear(self) -> None:
        with self._main_barrier:
            self._cache.clear()
            self._cache_barriers.clear()
            self._bar_threads.clear()
            self._n_misses = 0
            self._n_storage_misses = 0

    def start(self) -> None:
        # for copied StackCaches, whose threads are not started by default
        for t in self._bar_threads.values():
            if not t.is_alive():
                t.start()

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

    def save_stock(self, key: tuple[str, str], stk: Optional[Stock] = None) -> None:
        # if no storage is allowed
        if self.maxstorage <= 0:
            return

        self.save_path.mkdir(exist_ok=True)
        if stk is None:
            # we assume we are already under the main barrier
            with self._cache_barriers[key]:
                path = self._get_stk_path(key)
                self._cache[key].save(path)
        else:
            # if we specify already the stock, we don't need any locks
            path = self._get_stk_path(key)
            stk.save(path)

    def load_stock(self, key: tuple[str, str]) -> Optional[Stock]:
        # if no storage is allowed
        if self.maxstorage <= 0:
            return None

        # we assume we are already under the main barrier
        path = self._get_stk_path(key)
        stk = Stock.load(path)
        # stock is not in storage
        if stk is None:
            return None
        # remove stocks following LRU policy
        files = [f for f in self.save_path.glob('*.attrs.pkl')]
        N = len(files)
        if N > self.maxstorage:
            oldest_time = files[0].lstat().st_mtime
            oldest_file = files[0]
            for f in files[1:]:
                t = f.lstat().st_mtime
                if t < oldest_time:
                    oldest_time = t
                    oldest_file = f
            # delete oldest
            oldest_file.unlink(missing_ok=True)
            _LOGGER.info(f'Removed "{oldest_file}" from StockCache\'s storage.')
        _LOGGER.info(f'Loaded "{stk}" from StockCache\'s storage.')
        return stk

    def _get_stk_path(self, key: tuple[str, str]) -> str:
        return str(self.save_path.joinpath(f'{"_".join(key)}'))

    def _hist_update(self, bar: str, sleep_interval: float = 0.5) -> None:
        """Update all cached stocks with certain bar time-interval.

        This method is supposed to be executed in a thread using
        :class:`modules.utils.TracedThread`.
        """
        # adhere to the bar's time grid
        dt = utils.ibkr_to_timedelta(bar)
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
            period = utils.seconds_to_ibkr(secs)

            _validate_period_bar(period, bar)
            _update_stock(stk, period, bar)
            self.save_stock(key=(stk.symbol, bar), stk=stk)

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
                with ThreadPoolExecutor() as executor:
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
            # whether the stock is in cache
            if key in self._cache:
                with self._cache_barriers[key]:
                    stock = self._cache[key]
                # if stored stock's period is shorter, we need to update it
                if pd.Timestamp.today() - stock.hist.index[0] >= utils.ibkr_to_timedelta(period):
                    # end of critical section
                    self._main_barrier.release()
                    return stock
                else:
                    # do not request NEW stock! instead update the existing one
                    with self._cache_barriers[key]:
                        self._main_barrier.release()
                        # request new stock
                        new_stock = _request_stock(name, period, bar)
                        # set stock to newly gotten stock
                        stock.set(new_stock)
                    return stock
            # this condition should not be possibly true! (since we init the _cache to None)
            # elif key in self._cache_barriers:
                # warnings.warn(f'A race condition happened when requesting key "{key}"!')
                # ...
            else:
                # increment miss count
                self._n_misses += 1

                # create hist auto-update if it does not exist
                if bar not in self._bar_threads:
                    self._bar_threads[bar] = TracedThread(target=self._hist_update, args=(bar,))
                    self._bar_threads[bar].start()

                # create lock for cache entry
                self._cache_barriers[key] = threading.Lock()
                # immediately lock it
                self._cache_barriers[key].acquire()

                # we may also have the stock saved in storage.
                # this function also handles max storage limits
                stock = self.load_stock(key=key)
                # if it is NOT saved in storage
                if stock is None:
                    # increment storage miss count
                    self._n_storage_misses += 1

                    # Evict according to LRU policy.
                    # This is possible thanks to the order-preserving
                    # python dicts.
                    keys = list(self._cache.keys())
                    if len(keys) >= self.maxmem:
                        self._cache.pop(keys[0])
                    # temporarely add cache entry
                    self._cache[key] = None
                else:
                    # if it IS saved in storage
                    self._cache[key] = stock
                    # end of critical section
                    self._main_barrier.release()
                    self._cache_barriers[key].release()
                    return stock
        except BaseException as e:
            # it's important to release the lock, even with an exception ;)
            self._main_barrier.release()
            raise e

        # end of critical section
        self._main_barrier.release()

        try:
            # request new stock
            stock = _request_stock(name, period, bar)
            with self._main_barrier:
                self._cache[key] = stock
        except Exception as e:
            self._cache.pop(key)
            # unlock this stock's mutex
            self._cache_barriers[key].release()
            raise e

        # unlock this stock's mutex
        self._cache_barriers[key].release()

        # save in storage
        self.save_stock(key, stock)

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
    try:
        info_data[0]["conid"]
    except Exception:
        raise ValueError(f'Unexpected response. Data: {info_data}')
    hist_data = cmd.MarketDataHistory()(info_data[0]["conid"], period=period, bar=bar)
    return Stock(info_data, hist_data)


def _update_stock(stk: Stock, period: str, bar: str):
    hist_data = cmd.MarketDataHistory()(stk.conid, period=period, bar=bar)
    stk.insert(hist_data)


class _ServerState(object):
    """Server's state containing all information on instruments and positions.

    This class is used internally to help set a simulation context.
    """

    _stock_cache: 'StockCache'
    _portfolios: list[Portfolio]
    _simulated: bool

    def __init__(self, srv: Server) -> None:
        self._stock_cache = srv._stock_cache.copy()

        # do not copy again the stocks
        stocks = list(self._stock_cache._cache.values())

        self._portfolios = [pf.copy(stocks=stocks) for pf in srv._portfolios]
        self._simulated = srv._simulated


class _SimulatedContext(object):
    """Class used by :class:`modules.Server` to simulated orders."""
    _server: Server
    _state: _ServerState

    def __init__(self, srv: Server) -> None:
        self._server = srv

    def __enter__(self) -> None:
        if self._server._simulated:
            raise ValueError('You are already in a simulated context!')

        _LOGGER.warn('ENTERING SIMULATED CONTEXT. ALL FOLLOWING TRANSACTIONS WILL BE SIMULATED.')
        self._server._simulated = True
        for pf in self._server._portfolios:
            pf._simulated = True
        self._state = self._server.copy_state()

    def __exit__(self, type, value, tb) -> None:
        _LOGGER.warn('LEAVING SIMULATED CONTEXT. ALL FOLLOWING TRANSACTIONS WILL BE REAL.')
        self._server.set_state(self._state)
        self._server._simulated = False
        for pf in self._server._portfolios:
            pf._simulated = False
