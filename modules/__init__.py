import logging


def initialize():
    if '_ibkr_algotrading_initialized' in globals():
        return
    global _ibkr_algotrading_initialized
    _ibkr_algotrading_initialized = True

    logger = logging.getLogger('ibkr-algotrading')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    formatter = \
        logging.Formatter('[%(asctime)s] %(levelname).1s T%(thread)d %(filename)s:%(lineno)s: %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

initialize()
