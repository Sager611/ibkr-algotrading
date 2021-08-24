import sys
import logging

logging.basicConfig(stream=sys.stdout,
                    format='[%(asctime)s] %(levelname).1s T%(thread)d %(filename)s:%(lineno)s: %(message)s',
                    level=logging.INFO)
