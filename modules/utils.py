"""Utility methods and classes for global usage."""

import logging
import sys
import time
import trace
import threading

# IBKR's fraction of wealth that is taken as commission
IBKR_COMMISSION_FRACTION = 0.01

_LOGGER = logging.getLogger('ibkr-algotrading')


class TracedThread(threading.Thread):
    """This class allows threads which can be killed.

    Note that if the thread calls `time.sleep()`, it won't receive signals and
    thus cannot be stopped in the duration it is asleep.

    :arg timeout:
    :type timeout: float
    """
    def __init__(self, *args, timeout: float = None, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.killed = False
        self._start_t = time.time()
        self.timeout = timeout

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.timeout and time.time() - self._start_t >= self.timeout:
            self.killed = True
            _LOGGER.warn(f'Killed thread for exceeding its timeout of {self.timeout:.2f}s.')
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True


def get_adjusted_wealth(quantity: float):
    """Return the amount of wealth we can use in a single order so as to match :param:`quantity`.

    If we have 100. USD and want to use it all to buy and/or sell some stocks, we can't simply
    perform the transaction as we will spend 100. * (1. + COMMISSION) USD.
    This function returns the amount X required to spend X * (1 + COMMISSION) = 100. USD in
    the transactions.
    """
    return quantity / (1. + IBKR_COMMISSION_FRACTION)
