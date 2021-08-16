#!/bin/python3
"""Entry script."""

import time
from modules.algorithms import Hedge
from modules.server import Server

if __name__ == "__main__":
    srv = Server()
    srv.start()

    start_t = time.perf_counter()
    for stk in srv[['AAPL', 'SPY', 'MSFT', 'TSLA'], '2m', '1d']:
        print(stk)
        # print(stk.hist)
    print(f'Elapsed: {time.perf_counter() - start_t}s')

    start_t = time.perf_counter()
    for stk in srv[['AAPL', 'SPY', 'MSFT', 'TSLA'], '2m', '1d']:
        print(stk)
        # print(stk.hist)
    print(f'Elapsed: {time.perf_counter() - start_t}s')

    # hedge = Hedge(srv)
    # hedge.start()
