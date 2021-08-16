"""Contains classes defining the different financial instruments."""

import pandas as pd
from functools import cached_property


class BaseInstrument(object):
    conid: str
    symbol: str
    hist: pd.DataFrame

    def __str__(self) -> str:
        s = f'{self.__class__.__name__}("{self.symbol}", conid={self.conid})'
        return s


class Stock(BaseInstrument):
    """Stock."""
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

    def get_returns(self) -> pd.DataFrame:
        """Get returns in the bar of the stock (daily, hourly, etc.)."""
        df = pd.DataFrame(self.hist, columns=['returns'])
        close = self.hist['close']
        df['returns'][1:] = close[1:] / close[:-1].values - 1
        df['returns'][0] = 0
        return df

    @cached_property
    def bar(self) -> pd.Timestamp:
        """Represents the time step in the historical data."""
        if not isinstance(self.hist.index, pd.DatetimeIndex):
            raise TypeError('The index of the historical data is not in Datetime format. '
                            f'It is of type: {type(self.hist.index)}')
        return self.hist.index[1] - self.hist.index[0]

    def buy(self):
        raise NotImplementedError()

    def sell(self):
        raise NotImplementedError()


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
