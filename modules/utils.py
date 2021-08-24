"""Utility methods and classes for global usage."""

import logging
import sys
import time
import trace
import threading


class TracedThread(threading.Thread):
    """This class allows threads which can be killed.

    Note that if the thread calls `time.sleep()`, it won't receive signals and
    thus cannot be stopped in the duration it is asleep.

    :arg timeout:
    :type timeout: float
    """
    def __init__(self, *args, timeout: float = None, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
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
            logging.warn(f'Killed thread for exceeding its timeout of {self.timeout:.2f}s.')
        if self.killed:
            if event == 'line':
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True
