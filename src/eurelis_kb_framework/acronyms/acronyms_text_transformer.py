from eurelis_kb_framework.text_transformers import TextTransformer


class AcronymsTextTransformer(TextTransformer):
    """
    Acronyms text transformer
    """

    def __init__(self, acronyms: dict[str, str]):
        """
        Intializer

        Args:
            acronyms(dict): the map between acronyms and their signification
        """

        self._acronyms: dict[str, str] = acronyms

    def transform(self, text: str) -> str:
        if not text:
            return ""

        import re

        for key, value in self._acronyms.items():
            text = re.sub(rf"\b{key}\b", f"{key} ({value})", text)

        return text
