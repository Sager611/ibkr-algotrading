import time
from random import random

import pandas as pd
import numpy as np

from modules.instruments import Stock
from modules import utils

# -------------------------- UTILITY FUNCTIONS -------------------------- #

def create_random_stock(start_date=None) -> Stock:
    if start_date is None:
        start_date = pd.Timestamp.today() - pd.Timedelta(days=100)

    info_data = [
        {
            "conid": "756733",
            "symbol": "SPY",
            "companyHeader": "SPDR S&P 500 ETF TRUST - ARCA",
            "companyName": "SPDR S&P 500 ETF TRUST"
        }
    ]

    dates = pd.date_range(start=start_date, end=pd.Timestamp.today()).view(np.int64)

    hist_data = {
        "data": [
            {
                "o": random(),
                "c": random(),
                "h": random(),
                "l": random(),
                "v": int(random() * 10000),
                "t": date // 1e6
            }
            for date in dates
        ]
    }

    return Stock(info_data=info_data, hist_data=hist_data)

# -------------------------------- TESTS -------------------------------- #


def test_fill_like():
    stk1 = create_random_stock(start_date=pd.Timestamp.today()-pd.Timedelta(days=200))
    stk2 = create_random_stock()

    t1 = time.perf_counter()
    utils.fill_like(stk1.hist, stk2.hist)
    t2 = time.perf_counter()
    print(f'Performance: {t2-t1:.4g}s')
    assert t2-t1 <= 2e-1, 'Too slow!'
