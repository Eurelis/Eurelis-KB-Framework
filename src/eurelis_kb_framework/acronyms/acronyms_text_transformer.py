from eurelis_kb_framework.text_transformers import TextTransformer


class AcronymsTextTransformer(TextTransformer):
    def __init__(self, acronyms: dict[str, str]):
        self._acronyms: dict[str, str] = acronyms

    def transform(self, input: str) -> str:
        if not input:
            return ""

        import re

        for key, value in self._acronyms.items():
            input = re.sub(rf"\b{key}\b", f"{key} ({value})", input)

        return input
