from abc import ABC, abstractmethod
from typing import Sequence


class Output(ABC):
    """Base output class"""

    @abstractmethod
    def print(self, *args, **kwargs):
        pass

    @abstractmethod
    def critical_print(self, *args, **kwargs):
        pass

    @abstractmethod
    def verbose_print(self, *args, **kwargs):
        pass

    @abstractmethod
    def status(self, msg: str, handler):
        pass

    @abstractmethod
    def verbose_status(self, msg, handler):
        pass

    @abstractmethod
    def print_table(self, items, columns: Sequence[str], row_extractor, **kwargs):
        pass

    @abstractmethod
    def verbose_print_table(
        self, items, columns: Sequence[str], row_extractor, **kwargs
    ):
        pass
