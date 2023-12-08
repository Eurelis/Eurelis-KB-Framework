import logging
from enum import Enum
from typing import Union, Optional, TYPE_CHECKING

from eurelis_kb_framework.base_factory import BaseFactory
from eurelis_kb_framework.output.base_console_output import BaseConsoleOutput
from eurelis_kb_framework.output.output import Output

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class Verbosity(Enum):
    CONSOLE_INFO = "console-info"
    CONSOLE_DEBUG = "console-debug"
    LOG_INFO = "log-info"
    LOG_DEBUG = "log-debug"


VERBOSE_VALUE = Union[Optional[Union[bool]], Verbosity]


class OutputFactory(BaseFactory[Output]):
    """
    Console output factory
    """

    def __init__(self):
        self.verbosity_level: Verbosity = Verbosity.LOG_INFO

    def set_verbose(self, verbose: VERBOSE_VALUE):
        """
        setter for the verbose property
        Args:
            verbose (None, boolean, Verbosity): if None or True will use log_info, if False will use log_debug

        Returns:

        """
        if verbose is None:
            self.verbosity_level = Verbosity.LOG_INFO
        elif isinstance(verbose, bool):
            self.verbosity_level = (
                Verbosity.LOG_DEBUG if verbose else Verbosity.LOG_INFO
            )
        elif not isinstance(verbose, Verbosity):
            raise ValueError(
                f"Invalid verbose parameter type, expecting None, True, False or Verbosity enum value, got {type(verbose)}"
            )
        else:
            self.verbosity_level = verbose

    def build(self, context: "BaseContext") -> Output:
        """
        Method to construct a BaseConsoleOutput
        Args:
            context: context object, usually the current instance of langchain_wrapper

        Returns:
            instance of BaseConsoleOutput or of a class inheriting it
        """
        if self.verbosity_level == Verbosity.LOG_INFO:
            from eurelis_kb_framework.output.logging_console_output import (
                LoggingConsoleOutput,
            )

            return LoggingConsoleOutput(logging.INFO)
        elif self.verbosity_level == Verbosity.LOG_DEBUG:
            from eurelis_kb_framework.output.logging_console_output import (
                LoggingConsoleOutput,
            )

            return LoggingConsoleOutput(logging.DEBUG)
        elif self.verbosity_level == Verbosity.CONSOLE_DEBUG:
            from eurelis_kb_framework.output.verbose_console_output import (
                VerboseConsoleOutput,
            )

            return VerboseConsoleOutput()

        return BaseConsoleOutput()
