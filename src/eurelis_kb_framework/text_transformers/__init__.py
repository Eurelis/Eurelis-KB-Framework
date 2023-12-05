from abc import ABC, abstractmethod


class TextTransformer(ABC):
    """
    Base definition of a text transformer
    """

    @abstractmethod
    def transform(self, text: str) -> str:
        """Method to transform a text.

        Args:
            text (str): The text to transform.

        Returns:
            str, transformed text
        """
        pass
