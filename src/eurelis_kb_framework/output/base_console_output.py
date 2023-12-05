from typing import List

from rich.console import Console

from eurelis_kb_framework.output.output import Output


class BaseConsoleOutput(Output):
    """
    Base output class for non-verbose printing on the console
    """

    def __init__(self):
        """
        Constructor
        Args:
            console: riche.console.Console instance
        """
        self.console = Console()
        self.error_console = Console(stderr=True, style="bold red")

    def print(self, *args, **kwargs):
        """
        Print method, proxy for the console print method
        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:

        """
        self.console.print(*args, **kwargs)

    def critical_print(self, *args, **kwargs):
        """
        Print method, proxy for the error console print method
        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:

        """
        self.error_console.print(*args, **kwargs)

    def verbose_print(self, *args, **kwargs):
        """
        Verbose print method, does nothing in the base version
        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
        """

        pass

    def status(self, msg: str, handler):
        """
        Status method, proxy for the console status method
        Args:
            msg: text message to display
            handler: function or lambda without a parameter

        Returns:
            the result of the handler method

        """
        with self.console.status(msg):
            value = handler()

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
        return handler()

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
        from rich.table import Table

        # prepare the table
        table = Table(**kwargs)

        # add the columns
        for column in columns:
            table.add_column(column)

        # add the rows
        for index, item in enumerate(items):
            row = row_extractor(index, item)
            table.add_row(*row)

        self.console.print(table)

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
