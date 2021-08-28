"""Contains classes defining the different financial instruments."""

import concurrent
import logging
from functools import cached_property
from typing import Iterable, Optional, Union

import pandas as pd
import numpy as np

from . import commands as cmd
from . import utils

# IBKR's maximum allowed precision when buying/selling contracts
MIN_ORDER_AMOUNT_VARIATION = 0.0001

_LOGGER = logging.getLogger('ibkr-algotrading')


class BaseInstrument(object):
    """Instrument class for stocks, options and futures.

    WARNING: only stocks are supported.

    :param hist: historical data on the instrument.
        Please note that the index of this dataframe is Pandas' DatetimeIndex,
        and it is in UTC format.
    :type hist: :class:`pandas.DataFrame`
    """
    conid: str
    symbol: str
    hist: pd.DataFrame
    _server: 'Server'

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}("{self.symbol}", conid={self.conid})'
        return s


class Stock(BaseInstrument):
    """Stock class."""
    companyHeader: str
    companyName: str

    def __init__(self, info_data: list[dict], hist_data: dict) -> None:
        """Constructor.

        :param info_data: contains returned data by class:`commands.StockInfo`
        :type: dict
        :param hist_data: contains returned data by class:`commands.MarketDataHistory`
        :type: dict
        """
        super().__init__()
        self.conid = info_data[0]["conid"]
        self.symbol = info_data[0]["symbol"]
        self.companyHeader = info_data[0]["companyHeader"]
        self.companyName = info_data[0]["companyName"]

        # historical data
        self.hist = _hist_data_to_dataframe(hist_data)

    @cached_property
    def bar(self) -> pd.Timedelta:
        """Represents the time step in the historical data."""
        if not isinstance(self.hist.index, pd.DatetimeIndex):
            raise TypeError('The index of the historical data is not in Datetime format. '
                            f'It is of type: {type(self.hist.index)}')
        return (self.hist.index[1:] - self.hist.index[:-1]).min()

    @property
    def period(self) -> pd.Timedelta:
        """The time period in the historical data."""
        if not isinstance(self.hist.index, pd.DatetimeIndex):
            raise TypeError('The index of the historical data is not in Datetime format. '
                            f'It is of type: {type(self.hist.index)}')
        return self.hist.index[-1] - self.hist.index[0]

    def get_returns(self) -> pd.DataFrame:
        """Get returns in the bar of the stock (daily, hourly, etc.)."""
        df = pd.DataFrame(self.hist, columns=['returns'])
        close = self.hist['close']
        df['returns'][1:] = close[1:] / close[:-1].values - 1
        df['returns'][0] = 0
        return df

    def insert(self, hist_data: dict) -> None:
        """Append new historical data."""
        df = _hist_data_to_dataframe(hist_data)
        # this is the best way I found to union in pandas.
        # first, find set1 \ set2
        idx = df.index.difference(self.hist.index)
        # the behavior of copy=False is not documented (gotta love pandas).
        # hopefully, it uses less memory.
        self.hist = pd.concat([self.hist, df.loc[idx]], copy=False)
        # have to sort the DatetimeIndex to have it in order.
        # why like this? well, as of Pandas 1.2.3 the .loc[]
        # approach to add rows does not work with a Datetime index!
        self.hist.sort_index(inplace=True)

    def value(self) -> tuple[float, Optional[pd.Timestamp]]:
        """Return best-effort current value of the stock and its time.

        Please note that the returning time is in UTC format, not the system's datetime format.

        :return: the stock's value and its corresponding time. If time cannot for some
            reason be converted to a pandas Timestamp, this method returns the value and
            `None`.
        """
        market_data = cmd.MarketData()(conids=self.conid, fields=["31"])
        # 31: the last price at which the contract traded.
        value = float(market_data[0]["31"])
        time = pd.to_datetime(market_data[0]["_updated"], unit='ms')
        if type(time) is not pd.Timestamp:
            time = None
        return value, time

    def buy(self, amount: float, srv):
        raise NotImplementedError()

    def sell(self, amount: float, srv):
        raise NotImplementedError()

    def preview_buy(self, quantity: float, srv) -> tuple[float, float]:
        """Perform a preview buy order.

        :return: commission and stock share value.
        """
        if srv is None:
            raise ValueError('Server argument is required to perform buy preview order')

        if quantity % MIN_ORDER_AMOUNT_VARIATION != 0.0:
            _LOGGER.warn(f'Provided quantity for stock "{str(self)}" is too precise. '
                         f'The following amount will be lost: {quantity % MIN_ORDER_AMOUNT_VARIATION}')
            quantity = round(quantity / MIN_ORDER_AMOUNT_VARIATION) * MIN_ORDER_AMOUNT_VARIATION

        data = cmd.PreviewOrders()(
            stocks=[self.conid],
            side="BUY",
            quantity=float(quantity),
            accountId=srv.accountId
        )
        comm = float(data["amount"]["commission"].split()[0])
        stk_value = float(data["amount"]["amount"].split()[0])

        # check that total is correct
        total = float(data["amount"]["total"].split()[0])
        if abs(total - (stk_value + comm)) > 1e-6:
            raise ValueError('total != commission + stock. '
                             f'total: {total} | commission: {comm} | stock: {stk_value}. '
                             f'stock: {str(self)}')

        return comm, stk_value

    def preview_sell(self, quantity: float, srv) -> tuple[float, float]:
        """Perform a preview sell order.

        :return: commission and stock share value.
        """
        if srv is None:
            raise ValueError('Server argument is required to perform buy preview order')

        if quantity % MIN_ORDER_AMOUNT_VARIATION != 0.0:
            _LOGGER.warn(f'Provided quantity for stock "{str(self)}" is too precise. '
                         f'The following amount will be lost: {quantity % MIN_ORDER_AMOUNT_VARIATION}')
            quantity = round(quantity / MIN_ORDER_AMOUNT_VARIATION) * MIN_ORDER_AMOUNT_VARIATION

        data = cmd.PreviewOrders()(
            stocks=[self.conid],
            side="SELL",
            quantity=float(quantity),
            accountId=srv.accountId
        )
        comm = float(data["amount"]["commission"].split()[0])
        stk_value = float(data["amount"]["amount"].split()[0])

        # check that total is correct
        total = float(data["amount"]["total"].split()[0])
        if abs(total - (stk_value - comm)) > 1e-6:
            raise ValueError('total != stock - commission. '
                             f'total: {total} | commission: {comm} | stock: {stk_value}. '
                             f'stock: {str(self)}')

        return comm, stk_value


def _hist_data_to_dataframe(hist_data: dict) -> pd.DataFrame:
    """Prepare the input dict to a pandas DataFrame."""
    data = hist_data["data"]
    df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])

    unix_timestamps = [v['t'] for v in data]
    df['date'] = pd.to_datetime(unix_timestamps, unit='ms')

    df['open'] = [v['o'] for v in data]
    df['close'] = [v['c'] for v in data]
    df['high'] = [v['h'] for v in data]
    df['low'] = [v['l'] for v in data]
    df['volume'] = [v['v'] for v in data]

    # set the index
    df.set_index('date', inplace=True)

    return df


class Portfolio(object):
    """Portfolio class to handle the sell, buy and statistics of multiple stocks."""

    _initial_wealth: float
    _stocks: np.ndarray
    _alloc: np.ndarray
    _simulated: bool
    _commission: float
    server: 'Server'

    def __init__(self, wealth: float, stocks: Optional[Iterable[Stock]] = None) -> None:
        """Constructor."""
        self._initial_wealth = wealth
        self._stocks = np.empty(shape=(0,), dtype=Stock)
        self._alloc = np.empty(shape=(0,), dtype=float)
        self._simulated = False
        self._commission = 0.0

        if stocks is not None:
            self.insert(stocks)

    @property
    def wealth(self) -> float:
        """Wealth property is read-only."""
        # if there is no allocation, we still haven't
        # bought any stock
        if np.allclose(self._alloc, 0.0):
            return self._initial_wealth

        stocks_value = np.array([stk.value() for stk in self._stocks])
        # our wealth is decreased by the commissions we had during the
        # buy/sell transactions.
        return np.inner(stocks_value, self._alloc) - self._commission

    @property
    def stocks(self) -> np.ndarray:
        """Stocks property is read-only.

        :return: numpy array of stocks.
            Stock objects themselves are not copied.
        """
        return self._stocks.copy()

    @property
    def alloc(self) -> np.ndarray:
        """Allocation property is read-only.

        This array corresponds to the fraction of stock we are currently in possesion
        of, for each of the stocks.
        """
        return self._alloc.copy()

    @property
    def weights(self) -> np.ndarray:
        """Weights property is read-only.

        This array corresponds to the distribution of wealth accross the stocks.
        """
        stocks_value = np.array([stk.value()[0] for stk in self._stocks])
        return (self._alloc * stocks_value) / self.wealth

    def sharpe(self):
        """Calculate the Sharpe ratio of the portfolio."""
        raise NotImplementedError()

    def order(self, weights: np.ndarray) -> None:
        """Buy or sell stock in order to distribute the portfolio's wealth according to :param:`weights`."""
        if not hasattr(self, 'server'):
            raise RuntimeError('Cannot perform an order because the portfolio is not part of any server.')

        _validate_portfolio_weights(self._stocks, weights)
        stocks_value = np.array([stk.value()[0] for stk in self._stocks])

        # for the order, we have to take into account the commission taken by IBKR
        W = utils.get_adjusted_wealth(self.wealth)

        # how much stock we need for each stock
        new_alloc = weights * W / stocks_value
        # how much we have to buy/sell
        diff = new_alloc - self._alloc

        # the is a limited precision to the fraction of stock, or share, we can buy/sell.
        # we will adjust diff for that.
        diff = np.round(diff / MIN_ORDER_AMOUNT_VARIATION) * MIN_ORDER_AMOUNT_VARIATION

        # first, get an estimate for the commissions
        commissions, stocks_value_ = self.preview_order(diff)
        _LOGGER.info(f'weights: {weights} | diff: {diff} | comm: {commissions}')
        _LOGGER.info(f'stk vals: {stocks_value} | share vals: {stocks_value_}')

        # simulated context
        if self._simulated:
            _LOGGER.info('SIMULATED ORDER!!!!!!!')
            # for the simulation, we use the previewed order as if it were real
            self._alloc += diff
            # add to the commissions
            self._commission += commissions.sum()
        else:
            _LOGGER.info('REAL ORDER!!!!!!!')
            pass

    def preview_order(self, diff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Preview order.

        :param diff: how much stock to buy, if >0, or to sell, if <0.
        :type diff: numpy float array
        :return: commission for each of the buy/sell orders, and value of stock shares at the time of order.
        """
        if not hasattr(self, 'server'):
            raise RuntimeError('Cannot perform an order because the portfolio is not part of any server.')

        def _prev_order(args):
            d = args[0]
            stk = args[1]
            if d > 0:
                return stk.preview_buy(d, self.server)
            if d < 0:
                return stk.preview_sell(d, self.server)
            return None, None

        # perform requests in parallel
        commissions = []
        shares_values = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # async start requests
            futures = [
                executor.submit(_prev_order, (d, stk))
                for d, stk in zip(diff, self._stocks)
            ]
            # join threads
            for f in futures:
                c, s = f.result()
                if c is None:
                    continue
                commissions += [c]
                shares_values += [s]

        return np.array(commissions, dtype=float), np.array(shares_values, dtype=float)

    def insert(self, stocks: Union[Stock, Iterable[Stock]]) -> None:
        """Introduce new stock(s) to the portfolio."""
        if type(stocks) is Stock:
            stocks = [stocks]
        elif not hasattr(stocks, '__iter__'):
            raise TypeError(f'`stocks` is not iterable. It is of type: {type(stocks)}')
        else:
            # convert to list for easier usage
            stocks = [stk for stk in stocks]
        self._stocks = np.r_[self._stocks, stocks]
        self._alloc = np.r_[self._alloc, np.zeros(len(stocks))]


def _validate_portfolio_weights(stocks, weights):
    assert np.abs(weights.sum() - 1.0) < 1e-6, \
        f'Allocated wealth distribution does not sum to 1! Sums to: {weights.sum()}'
    assert len(stocks) == len(weights), \
        f'There is a different amount of stocks and weights! {len(stocks)} != {len(weights)}'
