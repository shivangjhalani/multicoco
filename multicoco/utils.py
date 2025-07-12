import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):

    def __init__(self, level: int=logging.NOTSET) -> None:
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)