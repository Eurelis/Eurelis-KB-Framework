import inspect
import logging
import sys
from typing import List


FORMAT = "%(asctime)s - %(name)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class LoggingConsoleOutput:
    """
    Base output class for non-verbose printing on the console
    """

    def __init__(self, console):
        """
        Constructor
        Args:
            console: riche.console.Console instance
        """
        self.console = console

    @property
    def logger(self) -> logging.Logger:
        return LoggingConsoleOutput._get_logger()

    @staticmethod
    def _get_logger() -> logging.Logger:
        frame = sys._getframe(3)
        module = inspect.getmodule(frame)
        return logging.getLogger(module.__name__)

    def print(self, *args, **kwargs):
        """
        Print method, proxy for the console print method
        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:

        """
        self.logger.info(*args, **kwargs)

    def verbose_print(self, *args, **kwargs):
        """
        Verbose print method, does nothing in the base version
        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
        """
        self.logger.debug(*args, **kwargs)

    def status(self, msg: str, handler):
        """
        Status method, proxy for the console status method
        Args:
            msg: text message to display
            handler: function or lambda without a parameter

        Returns:
            the result of the handler method

        """
        self.logger.info(f"[START] {msg}")
        value = handler()
        self.logger.info(f"[END] {msg}")

        return value

    def verbose_status(self, msg, handler):
        """
        Status method, proxy for the console status method, does nothing much in the base version
        Args:
            msg: text message to display
            handler: function or lambda without a parameter

        Returns:
            the result of the handler method

        """
        self.logger.debug(f"[START] {msg}")
        value = handler()
        self.logger.debug(f"[END] {msg}")

    def print_table(self, items, columns: List[str], row_extractor, **kwargs):
        """
        Print table method, will create a table and print it them
        Args:
            items: list or iterator for row items
            columns: list of columns names
            row_extractor: lambda method with two parameters (index and item) to extract row values
            kwargs: key value arguments to provide to the Table constructor

        Returns:

        """
        pass

    def verbose_print_table(self, items, columns: List[str], row_extractor, **kwargs):
        """
        Print table method, will create a table and print it them, does nothing in the base version
        Args:
            items: list or iterator for row items
            columns: list of columns names
            row_extractor: lambda method with two parameters (index and item) to extract row values
            kwargs: key value arguments to provide to the Table constructor

        Returns:

        """
        pass