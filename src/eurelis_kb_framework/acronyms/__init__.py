import json
from eurelis_kb_framework.acronyms.acronyms_text_transformer import (
    AcronymsTextTransformer,
)
from eurelis_kb_framework.base_factory import ParamsDictFactory
from eurelis_kb_framework.text_transformers import TextTransformer


class AcronymsTextTransformerFactory(ParamsDictFactory[TextTransformer]):
    def build(self, context: "BaseContext") -> TextTransformer:
        acronyms = None
        if "file" in self.params:
            with open(self.params.get("file")) as acronyms_fp:
                acronyms = json.load(acronyms_fp)

        elif "values" in self.params:
            acronyms = self.params.get("values")

        if not acronyms:
            raise ValueError(
                "Missing acronyms data, give either a file or values parameter"
            )

        if not isinstance(acronyms, dict):
            raise ValueError(f"Bad acronyms type, expected dict got {type(acronyms)}")

        return AcronymsTextTransformer(acronyms)
