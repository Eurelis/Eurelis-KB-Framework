from typing import List

from eurelis_kb_framework.output.base_console_output import BaseConsoleOutput


class VerboseConsoleOutput(BaseConsoleOutput):
    """
    Class for verbose printing on the console
    """

    def verbose_print(self, *args, **kwargs):
        """
        print method, proxy for the console print method
        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:

        """
        self.print(*args, **kwargs)

    def verbose_status(self, msg, handler):
        """
        status method, proxy for the console status method
        Args:
            msg: text message to display
            handler: function or lambda without a parameter

        Returns:
            the result of the handler method

        """
        return self.status(msg, handler)

    def verbose_print_table(self, items, columns: List[str], row_extractor, **kwargs):
        """
        Print table method, will create a table and print it them
        Args:
            items: list or iterator for row items
            columns: list of columns names
            row_extractor: lambda method with two parameters (index and item) to extract row values
            kwargs: key value arguments to provide to the Table constructor

        Returns:

        """
        self.print_table(items, columns, row_extractor, **kwargs)
