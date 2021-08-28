#!/bin/python3

import warnings
from typing import Union
from modules.commands import \
    AccountInfo, MarketData, PortfoliosInfo, \
    Balance, StockInfo, \
    PreviewOrders, MarketDataHistory


def menu():
    print('what do you wanna know?')
    print(
        '''
    a - Account information
    b - Balance
    d - Search Stock's latest market data
    h - Search Stock's history
    p - Portfolios information
    s - Search Stock information
    x - Preview orders
    q - Quit
        '''
    )
    return input('> ')


def pretty_print(arg: Union[dict, list], indent: int = 0) -> None:
    """Print a dictionary or list in a readable format.

    :param arg:
    :type arg: dict or list
    """
    if isinstance(arg, dict):
        for k, v in arg.items():
            print('\t' * indent + str(k) + ':')
            if isinstance(v, dict):
                pretty_print(v, indent+1)
            elif isinstance(v, list):
                pretty_print(v, indent)
            else:
                print('\t' * (indent+1) + str(v))
    elif isinstance(arg, list):
        print('\t' * indent + '[')
        for i in arg:
            if isinstance(i, dict):
                print('\t' * (indent+1) + '. ')
                pretty_print(i, indent+2)
            else:
                print('\t' * (indent+1) + '. ' + str(i))
        print('\t' * indent + ']')


def account_info():
    data = AccountInfo()()
    pretty_print(data)


def balance():
    p_data = PortfoliosInfo()()
    req_balance = Balance()
    for portfolio in p_data:
        print(portfolio["accountId"])
        print('-' * 10)
        data = req_balance(portfolio["accountId"])
        pretty_print(data)


_stocks_cache = {}
def stock_market_data():
    req_stock_info = StockInfo()
    print()
    stock = input('Provide stock name to preview order of (ex.: AAPL, MSFT, SPY): ')
    if stock in _stocks_cache:
        conid = _stocks_cache[stock][0]["conid"]
    else:
        conid = req_stock_info(stock)[0]["conid"]

    data = MarketData()(conids=conid)
    pretty_print(data)


def stock_history():
    req_stock_info = StockInfo()
    print()
    stock = input('Provide stock name to preview order of (ex.: AAPL, MSFT, SPY): ')
    if stock in _stocks_cache:
        conid = _stocks_cache[stock][0]["conid"]
    else:
        conid = req_stock_info(stock)[0]["conid"]
    period = input('period ({1-30}min, {1-8}h, {1-1000}d, {1-792}w, {1-182}m, {1-15}y) : ')
    bar = input('bar (blank for 1min) : ')
    if len(bar) == 0:
        bar = '1min'

    kwargs = {
        "conid": conid,
        "period": period,
        "bar": bar
    }

    data = MarketDataHistory()(**kwargs)
    pretty_print(data)


def portfolios_info():
    data = PortfoliosInfo()()
    pretty_print(data)


def preview_orders():
    req_stock_info = StockInfo()
    accountId = PortfoliosInfo()()[0]["accountId"]
    stock = ' '
    stocks, sides, quantity = [], [], []
    while stock != '':
        print()
        stock = input('Provide stock name to preview order of (blank to finish): ')
        if len(stock) == 0:
            break
        if stock in _stocks_cache:
            stocks += [_stocks_cache[stock][0]["conid"]]
        else:
            stocks += [req_stock_info(stock)[0]["conid"]]
        sides += [input('BUY / SELL : ')]
        quantity += [float(input('Quantity : '))]

    kwargs = {
        "stocks": stocks,
        "side": sides,
        "quantity": quantity,
        "accountId": accountId
    }

    data = PreviewOrders()(**kwargs)
    pretty_print(data)


def stock_info():
    print()
    stocks = input('Provide a space-separated list of stocks (ex.: AAPL, MSFT, SPY): ')
    req_stock_info = StockInfo()
    for stock in stocks.split(' '):
        if len(stock) == 0:
            continue
        print(stock)
        print('-' * 10)
        if stock in _stocks_cache:
            data = _stocks_cache[stock]
        else:
            data = req_stock_info(stock)
            _stocks_cache[stock] = data
        pretty_print(data)


if __name__ == "__main__":
    opts = ''
    while opts != 'q':
        opts = menu()
        for o in opts:
            try:
                if o == 'a':
                    account_info()
                if o == 'b':
                    balance()
                if o == 'd':
                    stock_market_data()
                if o == 'h':
                    stock_history()
                if o == 'p':
                    portfolios_info()
                if o == 's':
                    stock_info()
                if o == 'x':
                    preview_orders()
            except Exception as e:
                warnings.warn(f'Could not perform option "{o}". Exception: {str(e)}')
        opts = opts[-1]
