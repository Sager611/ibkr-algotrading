"""This module contains IBKR HTTP requests in the form of 'commands'."""

import re
import logging
import urllib3
import json
import time
from json.decoder import JSONDecodeError
from typing import Optional, Union
from pathlib import Path

_LOGGER = logging.getLogger('ibkr-algotrading')

if 'HTTP' not in globals():
    try:
        CACERT = Path(__file__).parent.parent.resolve().joinpath('container_inputs').joinpath('cacert.pem')
        if not CACERT.is_file():
            raise ValueError('Provide a certificate file saved at: ' + str(CACERT))
        CACERT = str(CACERT)
        HTTP = urllib3.PoolManager(cert_reqs='REQUIRED', ca_certs=CACERT)
    except Exception as e:
        # do not use certificates
        _LOGGER.warning(f'Could not use certificates! Exception: {e}')
        # TODO: this is a bad idea !
        HTTP = urllib3.PoolManager(cert_reqs='CERT_NONE',
                                   assert_hostname=False)


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
                extra_str = ''
                if r.status // 500 == 1:
                    extra_str = '(Internal server error)'
                _LOGGER.error(f'Non-json response: {str(r.data)} \n\t\t'
                              f'Status was: {r.status} {extra_str} \n\t\t'
                              f'Body was: {encoded_body}')
                return
        elif self.method == "GET":
            r = HTTP.request(self.method, url, headers=headers, fields=body_or_fields)
            try:
                ret = json.loads(r.data)
                return ret
            except JSONDecodeError:
                extra_str = ''
                if r.status // 500 == 1:
                    extra_str = '(Internal server error)'
                _LOGGER.error(f'Non-json response: {str(r.data)} \n\t\t'
                              f'Status was: {r.status} {extra_str} \n\t\t'
                              f'Fields were: {body_or_fields}')
                return
        else:
            raise TypeError(f'Method not supported: "{self.method}".')

class AccountInfo(Request):
    api_path = "/one/user"


class BrokerageAccounts(Request):
    """Returns a list of accounts the user has trading access to, their respective aliases and the currently selected account.

    Note this endpoint must be called before modifying an order or querying open orders.
    """

    api_path = "/iserver/accounts"


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


class MarketData(Request):
    """Get Market Data for the given conid(s).

    The endpoint will return by default bid, ask, last, change, change pct, close, listing exchange.
    See response fields for a list of available fields that can be request via fields argument.
    The endpoint /iserver/accounts must be called prior to /iserver/marketdata/snapshot.
    For derivative contracts the endpoint /iserver/secdef/search must be called first.
    First /snapshot endpoint call for given conid will initiate the market data request.
    To receive all available fields the /snapshot endpoint will need to be called several times.
    To receive streaming market data the endpoint /ws can be used. Refer to Streaming WebSocket Data for details.
    """
    api_path = "/iserver/marketdata/snapshot"
    LAST_PRICE_PATTERN = re.compile(r'(C|H)?([\d.]+)')

    @classmethod
    def last_price_to_float(cls, price: str) -> tuple[str, float]:
        """IBKR's field 31 has a particular format.

        31 format specification:
        The last price at which the contract traded.
        "C" identifies this price as the previous day's closing price.
        "H" means that the trading is halted.

        :return: type of price or empty string, and value of price
        :rtype: str, float
        """
        res = cls.LAST_PRICE_PATTERN.findall(price)
        if len(res) == 0:
            raise ValueError(
                f'Price is in invalid format: {price} \n'
                'Expected format: 20.0, C1.02, H321.1'
            )

        typ, val = res[0]
        val = float(val)
        return typ, val

    def __call__(self,
                 conids: Union[str, list[str]],
                 since: Optional[int] = None,
                 fields: Union[str, list[str]] = ["31", "83"]) -> Optional[Union[dict, list]]:
        """Execute request.

        By default it requests the following fields:
            31	string
            Last Price - The last price at which the contract traded.
            "C" identifies this price as the previous day's closing price.
            "H" means that the trading is halted.
            ----
            83 string
            Change % - The difference between the last price and
            the close on the previous trading day in percentage.
        """
        if type(conids) is list:
            conids_str = ','.join([str(i) for i in conids])
        else:
            conids_str = conids

        if type(fields) is list:
            fields_str = ','.join(fields)
        else:
            fields_str = fields

        req_fields = {
            "conids": conids_str,
            "fields": fields_str
        }

        if since is not None:
            req_fields["since"] = since

        data = super().__call__(req_fields)

        # if the fields are not in the data, we probably have to
        # first request our account information
        for i in range(10):
            repeat = False
            if data is None:
                repeat = True
            else:
                for f in fields:
                    if f not in data[0]:
                        repeat = True
                        break

            if repeat:
                _LOGGER.warning(f'Repeating market data request on conid(s) "{conids}" '
                                'since there wasn\'t an appropiate response. \n\t\t'
                                f'Response was: {data}')
                # for some reason IBKR asks us to request our accounts information
                # before requesting market data, o.w. we supposedly get a bad response.
                BrokerageAccounts()()
                # wait a bit before performing new request
                time.sleep(0.5)
                data = super().__call__(req_fields)
            else:
                break

        if i >= 9:
            _LOGGER.warning(f'Repeated market data request on conids "{conids}" too many times! (10)')

        return data


class MarketDataSingleCancel(Request):
    """Cancel market data for given conid."""

    "https://localhost:5000/v1/api/iserver/marketdata/{conid}/unsubscribe"
    api_path = None
    conid = None

    def set_conid(self, conid) -> None:
        self.conid = conid
        self.api_path = f'/iserver/marketdata/{conid}/unsubscribe'

    def __call__(self, conid=None, **kwargs) -> Optional[Union[dict, list]]:
        if self.api_path is None and conid is None:
            raise ValueError('Please provide portfolioId to check balance.')

        if conid is not None:
            self.set_conid(conid)

        return super().__call__(**kwargs)


class MarketDataAllCancel(Request):
    """Cancel all market data request(s)."""

    api_path = '/iserver/marketdata/unsubscribeall'


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


class TradingSchedule(Request):
    """Return the trading schedule up to a month for the requested contract."""

    api_path = "/trsrv/secdef/schedule"
    VALID_ASSETCLASSES = ["STK", "OPT", "FUT", "CFD", "WAR", "SWP", "FND", "BND", "ICS"]

    def __call__(self,
                 assetClass: str,
                 symbol: str,
                 exchange: Optional[str] = None,
                 exchangeFilter: Optional[str] = None) -> Optional[Union[dict, list]]:
        """

        :param assetClass: specify the asset class of the contract.
            Available values:
            Stock: STK, Option: OPT, Future: FUT, Contract For Difference: CFD,
            Warrant: WAR, Forex: SWP, Mutual Fund: FND, Bond: BND, Inter-Commodity Spreads: ICS
        """
        if assetClass not in TradingSchedule.VALID_ASSETCLASSES:
            raise ValueError(f'Unknown asset class: {assetClass}')

        fields = {
            "assetClass": assetClass,
            "symbol": symbol,
        }
        if exchange is not None:
            fields["exchange"] = exchange
        if exchangeFilter is not None:
            fields["exchangeFilter"] = exchangeFilter
        return super().__call__(fields)


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


class PreviewOrders(Request):
    api_path = None
    method = "POST"
    accountId = None

    def set_accountId(self, accountId: str):
        self.accountId = accountId
        self.api_path = f'/iserver/account/{accountId}/orders/whatif'

    def __call__(self,
                 stocks: list[str],
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

        for s in side:
            if s != "BUY" and s != "SELL":
                raise ValueError(f'Unrecognized side argument "{s}". Must be: BUY, SELL')

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
