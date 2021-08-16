"""Server module to handle requests independently of the computing algorithm(s)."""

import re
from functools import lru_cache

from . import commands as cmd
from typing import Iterable, Union
from .instruments import Stock


class Server(object):
    """Server class.

    It also safeguards against possibly unwanted orders, due to bugs or unexpected
    behavior of the algorithm(s) being used.
    """
    # _stock_cache: dict[str, Stock]

    def __init__(self) -> None:
        pass
        # self._stock_cache = {}

    def start(self) -> None:
        # TODO: prepary async threads the class might need
        pass

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

        # TODO: use multiprocessing for large lists to improve efficiency.
        stocks = []
        for stock_name in stock_names:
            stock = _request_stock(stock_name, period, bar)
            _validate_stock_equality(stock_name, period, bar, stock)
            stocks += [stock]

        if not was_stock_names_iterable:
            return stocks[0]
        return stocks

@lru_cache(maxsize=100, typed=False)
def _request_stock(stock_name: str, period: str, bar: str) -> Stock:
    """Return stock specified by the arguments and stores it following a LRU cache."""
    info_data = cmd.StockInfo()(stock_name)
    hist_data = cmd.MarketDataHistory()(info_data[0]["conid"], period=period, bar=bar)
    return Stock(info_data, hist_data)

def _validate_stock_equality(stock_name, period, bar, target_stock):
    """Check if the requested stock is in fact equivalent to the stock found."""
    if target_stock.symbol != stock_name:
        raise ValueError(f'Requested stock (name "{stock_name}") does not match target stock (name "{target_stock.name}").')
    # TODO: check period and bar
    # if target_stock.period != period:
        # raise ValueError(f'Requested stock (period "{period}") does not match target stock (period "{target_stock.period}").')
    # if target_stock.bar != bar:
        # raise ValueError(f'Requested stock (bar "{bar}") does not match target stock (bar "{target_stock.bar}").')


def _validate_period_bar(period, bar):
    """Check inputs are what IBKR is expecting."""
    res = _validate_period_bar.period_pattern.findall(period)
    if len(res) == 0:
        raise ValueError(
                f'Period is in invalid format: {period} \n'
                'Expected format: {1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y'
        )

    p_val, p_unit = res[0]
    res = _validate_period_bar.bar_pattern.findall(bar)
    if len(res) == 0:
        raise ValueError(
                f'bar is in invalid format: {bar} \n'
                'Expected format: 1min, 2min, 3min, 5min, 10min, 15min, 30min, 1h, 2h, 3h, 4h, 8h, 1d, 1w, 1m'
        )

    b_val, b_unit = res[0]

    unit_to_val = {'min': 0, 'h': 1, 'd': 2, 'w': 3, 'm': 4, 'y': 5}
    p_unit_ = unit_to_val[p_unit]
    b_unit_ = unit_to_val[b_unit]
    if b_unit_ > p_unit_ or (b_unit_ == p_unit_ and b_val > p_val):
        raise ValueError(f'Bar is larger than period: {bar} > {period}')


_validate_period_bar.period_pattern = re.compile(r'(\d+)(min|h|d|w|m|y)')
_validate_period_bar.bar_pattern = re.compile(r'(\d+)(min|h|d|w|m)')
