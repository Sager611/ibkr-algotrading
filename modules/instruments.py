"""Contains classes defining the different financial instruments."""

import concurrent
import logging
from functools import cached_property
from typing import Iterable, Optional, Union
from datetime import datetime

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
    # _server: 'Server'

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}("{self.symbol}", conid={self.conid})'
        return s


class Stock(BaseInstrument):
    """Stock class."""
    companyHeader: str
    companyName: str
    _info_data: list[dict]
    _hist_data: dict

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
        self._info_data = info_data
        self._hist_data = hist_data

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

    def copy(self) -> 'Stock':
        ins = Stock(self._info_data.copy(), self._hist_data.copy())
        ins.hist = self.hist.copy()
        return ins

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

    def preview_buy(self, quantity: float, srv) -> tuple[str, str, str]:
        """Perform a preview buy order.

        :return: commission, stock share value and total cost.
        """
        if srv is None:
            raise ValueError('Server argument is required to perform buy preview order')

        if quantity < MIN_ORDER_AMOUNT_VARIATION:
            raise ValueError(f'Quantity must be >={MIN_ORDER_AMOUNT_VARIATION}. It is: {quantity}')

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

        _validate_preview_order_data(data)

        comm = data["amount"]["commission"]
        share_val = data["amount"]["amount"]
        total = data["amount"]["total"]

        return comm, share_val, total

    def preview_sell(self, quantity: float, srv) -> tuple[str, str, str]:
        """Perform a preview sell order.

        :return: commission, stock share value and total cost.
        """
        if srv is None:
            raise ValueError('Server argument is required to perform buy preview order')

        if quantity < MIN_ORDER_AMOUNT_VARIATION:
            raise ValueError(f'Quantity must be >={MIN_ORDER_AMOUNT_VARIATION}. It is: {quantity}')

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

        _validate_preview_order_data(data)

        comm = data["amount"]["commission"]
        share_val = data["amount"]["amount"]
        total = data["amount"]["total"]

        return comm, share_val, total

    def __eq__(self, other) -> bool:
        if type(other) is Stock:
            return (self.bar == other.bar) and (self.conid == other.conid)
        return False


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
    _orders: pd.DataFrame
    # server: 'Server'

    def __init__(self, wealth: float, stocks: Optional[Iterable[Stock]] = None) -> None:
        """Constructor."""
        self._initial_wealth = wealth
        self._stocks = np.empty(shape=(0,), dtype=Stock)
        self._alloc = np.empty(shape=(0,), dtype=float)
        self._simulated = False
        self._commission = 0.0
        self._orders = pd.DataFrame(index=pd.DatetimeIndex([]))

        if stocks is not None:
            self.insert(stocks)

    @property
    def wealth(self) -> float:
        """Wealth property is read-only."""
        # if there is no allocation, we still haven't
        # bought any stock
        if np.allclose(self._alloc, 0.0):
            return self._initial_wealth

        stocks_value = np.array([stk.value()[0] for stk in self._stocks])
        return np.inner(stocks_value, self._alloc)

    @property
    def commission(self) -> float:
        """Return the total **estimated** commission that has decreased the wealth accross all orders.

        This attribute is read-only.
        """
        return self._commission

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

    @property
    def orders(self) -> pd.DataFrame:
        """Dataframe with the dates of execution for all orders and weights used.

        Orders property is read-only.
        """
        return self._orders.copy()

    def copy(self, stocks: Optional[Iterable[Stock]] = None) -> 'Portfolio':
        ins = Portfolio(self._initial_wealth, stocks)
        if hasattr(self, 'server'):
            ins.server = self.server
        if stocks is None:
            ins._stocks = np.array([stk.copy() for stk in self._stocks])
        else:
            # use the provided stocks as new target for copy
            ins._stocks = np.array([stk for stk in stocks if stk in self._stocks])
        ins._alloc = self._alloc.copy()

        if len(ins._stocks) != len(ins._alloc):
            raise ValueError('Provided stocks are incorrect. '
                             f'found {len(ins._stocks)} matching stocks when {len(ins._alloc)} were expected.')

        ins._simulated = self._simulated
        ins._commission = self._commission
        ins._orders = self._orders.copy()
        return ins

    def set(self, pf: 'Portfolio') -> None:
        if hasattr(self, 'server') and hasattr(self, 'server'):
            self.server = pf.server
        self._stocks = pf._stocks
        self._alloc = pf._alloc
        self._simulated = pf._simulated
        self._commission = pf._commission
        self._orders = pf._orders

    def sharpe(self):
        """Calculate the Sharpe ratio of the portfolio."""
        raise NotImplementedError()

    def order(self, weights: np.ndarray, date: Optional[pd.Timestamp] = None) -> None:
        """Buy or sell stock in order to distribute the portfolio's wealth according to :param:`weights`.

        :param date: Date at which to execute the order.
            This argument is used in a simulated context (see :meth:`modules.Server.simulated`) to
            retroactively perform an order. This helps when backtesting portfolio performance.
        :type date: pandas timestamp, optional
        """
        if not hasattr(self, 'server'):
            raise RuntimeError('Cannot perform an order because the portfolio is not part of any server.')

        if not self._simulated and date is not None:
            raise ValueError('You can only place an order back in time in a simulated context!')

        _validate_portfolio_weights(self._stocks, weights)

        # get stocks value today or at given date
        if date is None:
            stocks_value = np.array([stk.value()[0] for stk in self._stocks])
        else:
            stocks_value = []
            for stk in self._stocks:
                # first, check if we have to update the historical of the stock
                if date < stk.hist.index[0]:
                    today = datetime.today().timestamp()
                    period_secs = (today - stk.hist.index[-1].timestamp())
                    bar_secs = stk.bar.timestamp()

                    # if, somehow, the period is smaller that the bar
                    if period_secs < bar_secs:
                        period_secs = bar_secs

                    period = utils.seconds_to_ibkr(period_secs)
                    bar = utils.seconds_to_ibkr(bar_secs)

                    # by requesting the stock from the server,
                    # we are also updating its historical
                    self.server[stk.symbol, period, bar]

                # if the date is not stored in the stocks historical,
                # we cannot really perform an accurate simulated order
                if date not in stk.hist.index:
                    raise ValueError(f'date "{date}" is not in stock "{stk}" historical data.')

                stocks_value += [stk.hist["close"][date]]
            stocks_value = np.array(stocks_value)

        # for the order, we have to take into account the commission taken by IBKR
        prev_wealth = self.wealth
        W = utils.get_adjusted_wealth(prev_wealth)

        # how many shares we need for each stock
        new_alloc = weights * W / stocks_value
        # how much we have to buy/sell
        diff = new_alloc - self._alloc

        # the is a limited precision to the fraction of stock, or share, we can buy/sell.
        # we will adjust diff for that.
        diff = np.round(diff / MIN_ORDER_AMOUNT_VARIATION) * MIN_ORDER_AMOUNT_VARIATION

        # first, get an estimate for the commissions
        commissions, shares, total_spent = self.preview_order(diff)
        # TODO: DELETEME
        _LOGGER.info(f'weights: {weights} | diff: {diff} | comm: {commissions} | total spent: {total_spent}')
        _LOGGER.info(f'stk vals: {stocks_value} | share vals: {shares}')

        # simulated context
        if self._simulated:
            # for the simulation, we use the previewed order as if it were real
            self._alloc += diff
            # add to the commissions. this is an estimation for the actual commission.
            self._commission += prev_wealth - self.wealth
        else:
            # TODO: implement this
            _LOGGER.info('REAL ORDER!!!!!!!')
            raise NotImplementedError()

        # save date and weights of the order
        if date is None:
            order_date = pd.Timestamp.today()
        else:
            order_date = date
        self._save_order(order_date, weights)

    def _save_order(self, date, weights) -> None:
        # first, insert new stock columns
        symbols = [stk.symbol for stk in self._stocks]
        for sym in symbols:
            if sym in self._orders:
                continue
            # default weight value is 0, meaning the stock didn't have any allocated wealth
            self._orders.insert(len(self._orders.columns), column=sym, value=0)
        # now, introduce weights
        self._orders.loc[date, symbols] = weights

    def preview_order(self, diff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preview order.

        :param diff: how much stock to buy, if >0, or to sell, if <0.
        :type diff: numpy float array
        :return: commission for each of the buy/sell orders, value of stock shares at the time of order
            and total estimated cost.
        """
        if not hasattr(self, 'server'):
            raise RuntimeError('Cannot perform an order because the portfolio is not part of any server.')

        def _prev_order(args):
            d = args[0]
            stk = args[1]
            if d >= MIN_ORDER_AMOUNT_VARIATION:
                return stk.preview_buy(d, self.server)
            if d <= -MIN_ORDER_AMOUNT_VARIATION:
                return stk.preview_sell(-d, self.server)
            return None, None, None

        # perform requests in parallel
        commissions = []
        shares_values = []
        totals = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # async start requests
            futures = [
                executor.submit(_prev_order, (d, stk))
                for d, stk in zip(diff, self._stocks)
            ]
            # join threads
            for f, d in zip(futures, diff):
                c, s, t = f.result()
                if c is None:
                    continue
                commissions += [c]
                shares_values += [s]
                totals += [t]

        return np.array(commissions, dtype=str), \
            np.array(shares_values, dtype=str), \
            np.array(totals, dtype=str)

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

def _validate_preview_order_data(data):
    if type(data) is not dict:
        raise TypeError('Order data is not a dictionnary. \n\t\t'
                        f'data: {data}')
    if 'amount' not in data:
        raise ValueError(f'Order response is unexpected: {data}')
