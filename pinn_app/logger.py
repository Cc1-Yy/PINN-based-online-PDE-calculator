import sys
import logging
from typing import Any
from pinn_app.constants import LOG_BUFFER


class BufferHandler(logging.Handler):
    """
    Logging handler that appends formatted log messages to a shared buffer.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Format a log record and append it to LOG_BUFFER.

        :param record: The LogRecord instance containing event data
        """
        msg = self.format(record)
        LOG_BUFFER.append(msg)


class Tee:
    """
    A stream-like object that writes to an original stream and also logs each line.

    :param orig_stream: The original I/O stream (e.g., sys.__stdout__ or sys.__stderr__)
    :param logger: Logger instance for recording written data
    """

    def __init__(self, orig_stream: Any, logger: logging.Logger) -> None:
        self.orig = orig_stream
        self.logger = logger

    def write(self, data: str) -> None:
        """
        Write data to the original stream and log each line via the logger.

        :param data: The string data to write and log
        """
        # self.orig.write(data)
        for line in data.rstrip().splitlines():
            self.logger.info(line)

    def flush(self) -> None:
        """
        Flush the original stream.
        """
        # self.orig.flush()
        pass


def init_logger(name: str = 'pinn_logger') -> logging.Logger:
    """
    Create and configure a logger that writes INFO-level messages to LOG_BUFFER.

    :param name: Name of the logger to create or retrieve
    :return: Configured Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(message)s')
    buff = BufferHandler()
    buff.setFormatter(fmt)
    logger.addHandler(buff)

    return logger


def redirect_std_streams(logger: logging.Logger) -> None:
    """
    Redirect sys.stdout and sys.stderr so that printed output is also logged.

    :param logger: Logger instance to receive stdout and stderr writes
    """
    sys.stdout = Tee(sys.__stdout__, logger)
    sys.stderr = Tee(sys.__stderr__, logger)
