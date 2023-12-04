import json

from eurelis_kb_framework.acronyms.acronyms_text_transformer import (
    AcronymsTextTransformer,
)
from eurelis_kb_framework.base_factory import ParamsDictFactory
from eurelis_kb_framework.text_transformers import TextTransformer


class AcronymsTextTransformerFactory(ParamsDictFactory[TextTransformer]):
    """
    Factory for the acronyms text transformer
    """

    def build(self, context: "BaseContext") -> TextTransformer:
        """
        Construct the text transformer

        Args:
            context the context object, usually the current langchain wrapper instance

        Returns:
            a text transformer
        """
        acronyms = None
        # load from a file or a json object
        if "file" in self.params:
            with open(self.params.get("file")) as acronyms_fp:
                acronyms = json.load(acronyms_fp)

        elif "values" in self.params:
            acronyms = self.params.get("values")

        # we check we do have acronyms data
        if not acronyms:
            raise ValueError(
                "Missing acronyms data, give either a file or values parameter"
            )

        if not isinstance(acronyms, dict):
            raise ValueError(f"Bad acronyms type, expected dict got {type(acronyms)}")

        # we constrict the transformer
        return AcronymsTextTransformer(acronyms)
