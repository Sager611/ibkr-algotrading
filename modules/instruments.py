"""Contains classes defining the different financial instruments."""

import logging
import pickle
from functools import cached_property
from typing import Iterable, Optional, Union
from datetime import datetime
from pathlib import Path
from concurrent.futures.thread import ThreadPoolExecutor

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
        _validate_hist_data(hist_data)
        self.hist = _hist_data_to_dataframe(hist_data)

    @classmethod
    def load(cls, path: str) -> Optional['Stock']:
        # if it is not saved
        if not Path(path + '.attrs.pkl').exists():
            return None

        with open(path + '.attrs.pkl', 'rb') as f:
            attrs = pickle.load(f)
            _info_data = attrs.pop("_info_data")
            _hist_data = attrs.pop("_hist_data")
            ins = Stock(info_data=_info_data, hist_data=_hist_data)
            ins.conid = attrs.pop("conid")
            ins.symbol = attrs.pop("symbol")
            ins.companyHeader = attrs.pop("companyHeader")
            ins.companyName = attrs.pop("companyName")

        ins.hist = pd.read_pickle(path + '.hist.pkl')
        return ins

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

    def set_hist(self, hist: pd.DataFrame) -> None:
        """Set historical data, just in case :param:`hist` becomes read-only in the future."""
        self.hist = hist

    def copy(self) -> 'Stock':
        ins = Stock(self._info_data.copy(), self._hist_data.copy())
        ins.hist = self.hist.copy()
        return ins

    def set(self, stk: 'Stock') -> None:
        self.conid = stk.conid
        self.symbol = stk.symbol
        self.companyHeader = stk.companyHeader
        self.companyName = stk.companyName
        self._info_data = stk._info_data
        self._hist_data = stk._hist_data

        self.hist = stk.hist

    def save(self, path: str) -> None:
        attrs = {
            "conid": self.conid,
            "symbol": self.symbol,
            "companyHeader": self.companyHeader,
            "companyName": self.companyName,
            "_info_data": self._info_data,
            "_hist_data": self._hist_data
        }
        with open(path + '.attrs.pkl', 'wb') as f:
            pickle.dump(attrs, f)

        self.hist.to_pickle(path + '.hist.pkl')

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

    def get_closest_date(self, date: pd.Timestamp) -> pd.Timestamp:
        return self.hist.index[self.hist.index.get_loc(date, method='ffill')]

    def value(self, date: Optional[pd.Timestamp] = None) -> tuple[float, Optional[pd.Timestamp]]:
        """Return best-effort current value of the stock and its time.

        Please note that the returning time is in UTC format, not the system's datetime format.

        If you provide a date, the function will return the close value of the stock
        at the closest historical data date, that is also older than the provided date.

        :return: the stock's value and its corresponding time. If time cannot for some
            reason be converted to a pandas Timestamp, this method returns the value and
            `None`.
        """
        if date is None:
            market_data = cmd.MarketData()(conids=self.conid, fields=["31"])
            if "31" not in market_data[0]:
                raise ValueError(f'Response market data is in invalid format. Data: {market_data}')
            # 31: the last price at which the contract traded.
            _, value = cmd.MarketData.last_price_to_float(market_data[0]["31"])
            if "_updated" in market_data[0]:
                time = pd.to_datetime(market_data[0]["_updated"], unit='ms')
                if type(time) is not pd.Timestamp:
                    time = None
            else:
                time = None
            return value, time
        else:
            if date < self.hist.index[0]:
                raise ValueError(f'Date "{date}" is too old. Oldest stored date is: {self.hist.index[0]}')

            # get closest date in the dataframe that is still older than the given date
            closest_date = self.get_closest_date(date)

            return float(self.hist["close"][closest_date]), closest_date

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

        if quantity % MIN_ORDER_AMOUNT_VARIATION > 1e-8:
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
        # we implement this magic method so the if .. in .. condition works as we'd like
        if type(other) is Stock:
            return (self.bar == other.bar) and (self.conid == other.conid)
        return False


def _validate_hist_data(hist_data: dict) -> None:
    if type(hist_data) is not dict:
        raise TypeError(f'Historical data is not a dict. It is of type: {type(hist_data)}')

    if "data" not in hist_data:
        raise ValueError('Historical data is in an incorrect format. \n\t\t'
                         f'Data: {hist_data}')

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
    """Portfolio class to handle the sell, buy and statistics of multiple stocks.

    TODO: take dividends into account
    """

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
    def shares(self) -> np.ndarray:
        """Shares property is read-only.

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
        return (self._alloc * stocks_value) / self.wealth()

    @property
    def orders(self) -> pd.DataFrame:
        """Dataframe with the dates of execution for all orders and shares bought.

        Orders property is read-only.
        """
        return self._orders.copy()

    def wealth(self, date: Optional[pd.Timestamp] = None) -> float:
        """Amount of currency allocated to stock shares, or initial wealth if portfolio is empty.

        :param date: indicates the date at which we want to retrieve our wealth.
            If `None`, function returns a best-effort attempt at the latest portfolio wealth.
        :type date: pandas timestamp
        """
        # if there are no orders, we still haven't
        # bought any stock
        if len(self._orders) == 0:
            return self._initial_wealth

        if date is None:
            # current allocated shares
            shares = self._alloc
        else:
            # allocated shares at "date"
            i = self._orders.index.get_loc(date, method='ffill')
            shares = self._orders.iloc[i].values

        stocks_value = self.get_stocks_value(date=date)

        return np.inner(stocks_value, shares)

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

    def get_returns(self, end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Provide portfolio's returns in time."""
        if len(self._stocks) == 0:
            raise ValueError('Cannot calculate returns because there are no stocks in the portfolio')

        bar = self._stocks[0].bar
        for stk in self._stocks[1:]:
            if stk.bar != bar:
                raise ValueError('Cannot calculate returns because there are different bars'
                                 f'{self._stocks[0]} has bar "{bar}" while {stk} has bar "{stk.bar}"')

        # if end_date is not provided, we use the closest one to today
        # from stocks' historical
        if end_date is None:
            end_date = self._stocks[0].hist.index[-1]
        else:
            end_date = self._stocks[0].get_closest_date(end_date)

        order_dates = [
            self._stocks[0].get_closest_date(d) for d in self._orders.index if d <= end_date
        ]

        returns = pd.DataFrame(index=pd.DatetimeIndex([]), columns=['returns'])
        prev_val = 0.0
        # _orders index is assumed to be in chronological order
        for i, order_date in enumerate(order_dates):
            if i < len(self._orders) - 1:
                next_date = self._orders.index[i+1] - bar
            else:
                next_date = end_date
            if order_date > next_date:
                raise ValueError('Saved orders are not in chronological order: '
                                 f'{order_date} > {next_date} i: {i}')

            ref = self._stocks[0].hist[order_date:next_date]
            stocks_value = np.stack(
                [utils.fill_like(stk.hist[order_date:next_date], ref)['close'].values
                 for stk in self._stocks]
            ).T
            shares = self._orders.iloc[i].values[np.newaxis, :]
            # portfolio value by date
            port_val = (stocks_value * shares).sum(axis=1)
            rets = np.empty_like(port_val)
            rets[1:] = port_val[1:] / port_val[:-1] - 1

            # we have to take into account the returns accross orders as well
            if i > 0:
                rets[0] = port_val[0] / prev_val - 1
            prev_val = port_val[-1]

            idx = ref.index
            # for the first order, we ignore the 0 return, since just in that moment we put our
            # portfolio's wealth into the market for the first time
            if i == 0:
                idx = idx[1:]
                rets = rets[1:]

            rets = pd.DataFrame(rets, index=idx, columns=['returns'])

            returns = pd.concat([returns, rets])

        return returns

    def sharpe(self, end_date: Optional[pd.Timestamp] = None) -> float:
        """Calculate the Sharpe ratio of the portfolio.

        Computes the following formula:
            S = E[R_p - R_f] / std[R_p - R_f]
        where R_p are the returns and R_f are the risk-free rates
        """
        if not hasattr(self, 'server'):
            raise RuntimeError('Cannot calculate Sharpe ratio because the portfolio is not part of any server.')

        if len(self._stocks) == 0:
            raise ValueError('Cannot calculate Sharpe ratio because there are no stocks in the portfolio')

        bar = self._stocks[0].bar
        for stk in self._stocks[1:]:
            if stk.bar != bar:
                raise ValueError('Cannot calculate Sharpe ratio because there are different bars'
                                 f'{self._stocks[0]} has bar "{bar}" while {stk} has bar "{stk.bar}"')

        R_p = self.get_returns(end_date=end_date)

        # retrieve risk-free rates
        R_f = [
            self.server.risk_free_rot(date=date, dt=bar)[0] for date in R_p.index
        ]
        # to numpy
        R_p = R_p['returns'].values

        # factor so sharpe ratio is the same no matter the bar.
        # we assume a 252 trading year
        K = np.sqrt(pd.Timedelta(days=252) / bar)

        return K * np.mean(R_p - R_f) / np.std(R_p - R_f)

    def get_stocks_value(self, date: Optional[pd.Timestamp] = None) -> np.ndarray:
        """Get portfolio's stocks' value today, or at a given date."""
        if date is None:
            # TODO: make this parallel
            stocks_value = np.array([stk.value()[0] for stk in self._stocks])
        else:
            stocks_value = []
            for stk in self._stocks:
                # first, check if we have to update the historical of the stock
                if date < stk.hist.index[0]:
                    today = datetime.today().timestamp()
                    bar_secs = stk.bar.total_seconds()
                    period_secs = bar_secs + (today - stk.hist.index[0].timestamp())

                    bar = utils.seconds_to_ibkr(bar_secs)
                    period = utils.seconds_to_ibkr(period_secs)

                    # TODO: make this parallel accross stocks.
                    # by requesting the stock from the server,
                    # we are also updating its historical
                    self.server[stk.symbol, period, bar]

                # get closest date in the dataframe that is still older than the given date
                closest_date = stk.hist.index[stk.hist.index.get_loc(date, method='ffill')]

                stocks_value += [stk.hist["close"][closest_date]]
            stocks_value = np.array(stocks_value)
        return stocks_value

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

        if date is not None and len(self._orders) > 0 and  date < self._orders.index[-1]:
            raise ValueError('You have to perform orders in chronological order. '
                             f'Provided date: {date} | Last order\'s date: {self._orders.index[-1]}')

        _validate_portfolio_weights(self._stocks, weights)

        # get stocks value today or at given date
        stocks_value = self.get_stocks_value(date)

        # wealth that we have at our disposition
        W = self.wealth(date)

        adj_W = W
        comm = 0.0
        diff = np.zeros_like(self._alloc)

        for _ in range(10):
            prev_adj_W = adj_W
            # how many shares we need for each stock
            new_alloc = weights * adj_W / stocks_value
            # how much we have to buy/sell
            diff = new_alloc - self._alloc

            # there is a limited precision to the fraction of stock, or share, we can buy/sell.
            # we will adjust diff for that.
            diff = np.round(diff / MIN_ORDER_AMOUNT_VARIATION) * MIN_ORDER_AMOUNT_VARIATION

            # for the order, we have to take into account the commission taken by IBKR
            comm = utils.get_transaction_commission(weights * adj_W, diff, self._stocks, self.server)

            # update how much wealth we are actually have to transact so that
            # (trans. cost + commission) = wealth
            adj_W = W - comm
            if abs(adj_W - prev_adj_W) < 1e-6:
                break

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
            self._commission += comm
        else:
            # TODO: implement this
            _LOGGER.info('REAL ORDER!!!!!!!')
            raise NotImplementedError()

        # save date and weights of the order
        if date is None:
            order_date = pd.Timestamp.today()
        else:
            order_date = date
        self._save_order(order_date)

    def _save_order(self, date) -> None:
        # first, insert new stock columns
        symbols = [stk.symbol for stk in self._stocks]
        for sym in symbols:
            if sym in self._orders:
                continue
            # default weight value is 0, meaning the stock didn't have any allocated wealth
            self._orders.insert(len(self._orders.columns), column=sym, value=0)
        # now, introduce shares
        self._orders.loc[date, symbols] = self._alloc

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
        with ThreadPoolExecutor() as executor:
            # async start requests
            futures = [
                executor.submit(_prev_order, (d, stk))
                for d, stk in zip(diff, self._stocks)
            ]
            # join threads
            for f in futures:
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
    assert (weights >= 0).all(), \
        f'Negative weights are not allowed. weights: {weights}'

def _validate_preview_order_data(data):
    if type(data) is not dict:
        raise TypeError('Order data is not a dictionnary. \n\t\t'
                        f'data: {data}')
    if 'amount' not in data:
        raise ValueError(f'Order response is unexpected: {data}')
