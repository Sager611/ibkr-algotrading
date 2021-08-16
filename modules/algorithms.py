"""Contains meta-algorithms which exploit bagging of other lower level algorithms ('experts')."""

from .server import Server


class Hedge(object):
    """Hedge algorithm."""

    def __init__(self, srv: Server) -> None:
        self.server = srv

    def start(self) -> None:
        raise NotImplementedError()
