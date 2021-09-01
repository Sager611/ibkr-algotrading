#!/bin/python3
"""Entry script."""

from modules.server import Server
from modules.algorithms import Hedge

if __name__ == "__main__":
    srv = Server()
    srv.start()

    hedge = Hedge(srv)
    hedge.start()

    #
