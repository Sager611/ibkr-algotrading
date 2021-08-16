"""This module contains IBKR HTTP requests in the form of 'commands'."""

import warnings
import urllib3
import json
from json.decoder import JSONDecodeError
from typing import Optional, Union
from pathlib import Path

CACERT = Path(__file__).parent.parent.resolve().joinpath('container_inputs').joinpath('cacert.pem')
if not CACERT.is_file():
    raise ValueError('Provide a certificate file saved at: ' + str(CACERT))
CACERT = str(CACERT)
HTTP = urllib3.PoolManager(cert_reqs='REQUIRED', ca_certs=CACERT)


class Request():
    port: int
    api_path: Optional[str] = None
    method: str = "GET"

    def __init__(self, port: int = 5000) -> None:
        self.port = port

    def __call__(self, body_or_fields: Optional[Union[dict, list]] = None) -> Optional[Union[dict, list]]:
        if self.api_path is None:
            raise ValueError('\'Request\' is an abstract class.')

        url = f"https://localhost:{self.port}/v1/api" + self.api_path

        headers = {
            'User-Agent': 'ibeam/0.1.0',
            'Content-Type': 'application/json'
        }
        if self.method == "POST":
            encoded_body = json.dumps(body_or_fields)
            r = HTTP.urlopen(self.method, url, headers=headers, body=encoded_body)
            try:
                ret = json.loads(r.data)
                return ret
            except JSONDecodeError:
                warnings.warn('Non-json content: ' + str(r.data))
                return
        elif self.method == "GET":
            r = HTTP.request(self.method, url, headers=headers, fields=body_or_fields)
            try:
                ret = json.loads(r.data)
                return ret
            except JSONDecodeError:
                warnings.warn('Non-json content: ' + str(r.data))
                return
        else:
            raise TypeError(f'Method not supported: "{self.method}".')

class AccountInfo(Request):
    api_path = "/one/user"


class PortfoliosInfo(Request):
    api_path = "/portfolio/accounts"


class Balance(Request):
    api_path = None
    portfolioId = None

    def set_portfolioId(self, portfolioId) -> None:
        self.portfolioId = portfolioId
        self.api_path = f'/portfolio/{portfolioId}/ledger'

    def __call__(self, portfolioId=None, **kwargs) -> Optional[Union[dict, list]]:
        if self.api_path is None and portfolioId is None:
            raise ValueError('Please provide portfolioId to check balance.')

        if portfolioId is not None:
            self.set_portfolioId(portfolioId)

        return super().__call__(**kwargs)


class MarketDataHistory(Request):
    """Get historical market Data for given conid, length of data is controlled by 'period' and 'bar'.

    Formatted as: min=minute, h=hour, d=day, w=week, m=month, y=year e.g. period =1y with bar =1w returns 52 data points (Max of 1000 data points supported).
    Note: There's a limit of 5 concurrent requests. Excessive requests will return a 'Too many requests' status 429 response.
    """
    api_path = "/iserver/marketdata/history"

    def __call__(self, conid: str, exchange: str = '', period: str = '30min', bar: str = '1min', outsideRth: bool = False) -> Optional[Union[dict, list]]:
        fields = {
            "conid": conid,
            "exchange": exchange,
            "period": period,
            "bar": bar,
            "outsideRth": outsideRth
        }
        return super().__call__(fields)


class StockInfo(Request):
    api_path = "/iserver/secdef/search"
    method = "POST"

    def __call__(self, symbol: str, by_name: bool = False, secType: str = "STK") -> Optional[Union[dict, list]]:
        body = {
            "symbol": symbol,
            "name": by_name,
            "secType": secType
        }
        return super().__call__(body)


class PreviewOrders(Request):
    api_path = None
    method = "POST"
    accountId = None

    def set_accountId(self, accountId: str):
        self.accountId = accountId
        self.api_path = f'/iserver/account/{accountId}/orders/whatif'

    def __call__(self,
                 stocks: list[int],
                 secType: Union[str, list[str]] = "STK",
                 orderType: Union[str, list[str]] = "MKT",
                 side: Union[str, list[str]] = "BUY",
                 quantity: Union[int, float, list] = 0,
                 tif: Union[str, list[str]] = "GTC",
                 accountId: Optional[str] = None,
                 ) -> Optional[Union[dict, list]]:
        if self.accountId is None and accountId is None:
            raise ValueError('Please provide an accountId.')

        if accountId is not None:
            self.set_accountId(accountId)

        def _check_arg(arg) -> list:
            if type(arg) is str or type(arg) is int or type(arg) is float:
                arg = [arg] * len(stocks)
            else:
                assert len(arg) == len(stocks), f'`stocks` and `{arg.__name__}` must be the same length.'
            return arg

        secType = _check_arg(secType)
        orderType = _check_arg(orderType)
        side = _check_arg(side)
        quantity = _check_arg(quantity)
        tif = _check_arg(tif)

        body = {
            "orders": [
                {
                    "acctId": self.accountId,
                    "conid": int(stock),
                    "secType": f"{stock}:{secType[i]}",
                    "orderType": orderType[i],
                    "side": side[i],
                    "quantity": quantity[i],
                    "tif": tif[i]
                } for i, stock in enumerate(stocks)
            ]
        }
        return super().__call__(body)
