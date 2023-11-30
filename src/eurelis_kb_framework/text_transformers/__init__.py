from abc import ABC, abstractmethod


class TextTransformer(ABC):
    """
    Base definition of a text transformer
    """

    @abstractmethod
    def transform(self, input: str) -> str:
        """

        Args:
            input: text input

        Returns:
            str, transformed text
        """
        pass
