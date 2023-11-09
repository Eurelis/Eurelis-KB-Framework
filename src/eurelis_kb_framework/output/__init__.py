from rich.console import Console

from eurelis_kb_framework.base_factory import BaseFactory
from eurelis_kb_framework.output.base_console_output import BaseConsoleOutput


class ConsoleOutputFactory(BaseFactory[BaseConsoleOutput]):
    """
    Console output factory
    """

    def __init__(self):
        self.is_verbose = False

    def set_verbose(self, verbose: bool):
        """
        setter for the verbose property
        Args:
            verbose: boolean value

        Returns:

        """

        self.is_verbose = verbose

    def build(self, context) -> BaseConsoleOutput:
        """
        Method to construct a BaseConsoleOutput
        Args:
            context:

        Returns:

        """
        rich_console = Console()

        if self.is_verbose:
            from eurelis_kb_framework.output.verbose_console_output import (
                VerboseConsoleOutput,
            )

            return VerboseConsoleOutput(rich_console)
        if self.is_verbose is None:
            from eurelis_kb_framework.output.logging_console_output import (
                LoggingConsoleOutput,
            )

            return LoggingConsoleOutput(rich_console)

        return BaseConsoleOutput(rich_console)
