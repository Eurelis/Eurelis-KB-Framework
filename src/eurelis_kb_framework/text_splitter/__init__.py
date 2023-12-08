from typing import Tuple, cast, TYPE_CHECKING

from langchain.text_splitter import TextSplitter

from eurelis_kb_framework.base_factory import ParamsDictFactory, PARAMS

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext

TEXT_SPLITTER_ALLOWED_PARAMS = {
    "chunk_size",
    "chunk_overlap",
    "keep_separator",
    "add_start_index",
    "strip_whitespace",
}

CHARACTER_TEXT_SPLITTER_ALLOWED_PARAMS = {"separator", "is_separator_regex"}

TOKEN_TEXT_SPLITTER_ALLOWED_PARAMS = {
    "encoding_name",
    "model_name",
    "allowed_special",
    "disallowed_special",
}

SENTENCE_TRANSFORMER_TOKEN_TEXT_SPLITTER_ALLOWED_PARAMS = {
    "model_name",
    "tokens_per_chunk",
}

RECURSIVE_CHARACTER_TEXT_SPLITER_ALLOWED_PARAMS = {
    "separators",
    "keep_separator",
    "is_separator_regex",
}

NLTK_TEXT_SPLITTER_ALLOWED_PARAMS = {"separator", "language"}

SPACY_TEXT_SPLITTER_ALLOWED_PARAMS = {"separator", "pipeline"}

TEXT_SPLITTER_ALLOWED_TYPES = {
    "character": ("CharacterTextSplitter", CHARACTER_TEXT_SPLITTER_ALLOWED_PARAMS),
    "token": ("TokenTextSplitter", TOKEN_TEXT_SPLITTER_ALLOWED_PARAMS),
    "sentence-transformers-token": (
        "SentenceTransformersTokenTextSplitter",
        SENTENCE_TRANSFORMER_TOKEN_TEXT_SPLITTER_ALLOWED_PARAMS,
    ),
    "recursive-character": (
        "RecursiveCharacterTextSplitter",
        RECURSIVE_CHARACTER_TEXT_SPLITER_ALLOWED_PARAMS,
    ),
    "nltk": ("NLTKTextSplitter", NLTK_TEXT_SPLITTER_ALLOWED_PARAMS),
    "spacy": ("SpacyTextSplitter", SPACY_TEXT_SPLITTER_ALLOWED_PARAMS),
    "python-code": (
        "PythonCodeTextSplitter",
        RECURSIVE_CHARACTER_TEXT_SPLITER_ALLOWED_PARAMS,
    ),
    "markdown": (
        "MarkdownTextSplitter",
        RECURSIVE_CHARACTER_TEXT_SPLITER_ALLOWED_PARAMS,
    ),
    "latex": ("LatexTextSplitter", RECURSIVE_CHARACTER_TEXT_SPLITER_ALLOWED_PARAMS),
}


class GenericTextSplitterFactory(ParamsDictFactory[TextSplitter]):
    """
    Generic text splitter factory
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.provider = None

    def set_provider(self, provider: str):
        """Setter for the provider to use

        Args:
            provider (str): name of the provider to use

        Returns:

        """
        self.provider = provider

        if provider not in TEXT_SPLITTER_ALLOWED_TYPES:
            raise ValueError(
                f"Invalid text splitter {provider} provider, use one of {TEXT_SPLITTER_ALLOWED_TYPES.keys()}"
            )

    def _extract_arguments(self) -> Tuple[str, PARAMS]:
        """Helper method to extract arguments for the given provider

        Returns:
            Tuple[str, params]: provider name, params for the given provider

        """
        if not self.provider:
            raise ValueError(f"Missing provider parameter")

        provider_data = TEXT_SPLITTER_ALLOWED_TYPES.get(self.provider)

        provider_data_set: set[str] = cast(set[str], provider_data[1])

        splitter_arguments = self.extract_params(
            provider_data_set | TEXT_SPLITTER_ALLOWED_PARAMS
        )

        return provider_data[0], splitter_arguments

    def build(self, context: "BaseContext") -> TextSplitter:
        """
        Method to get the TextSplitter from
        Args:
            context: context object, usually the current instance of langchain_wrapper

        Returns:
            text splitter instance
        """
        (
            class_name,
            arguments,
        ) = self._extract_arguments()  # first to ensure a provider has been given

        instantiate_params = {"class": class_name, "kwargs": arguments}

        return cast(
            TextSplitter,
            context.loader.instantiate_class(
                "langchain.text_splitter", instantiate_params
            ),
        )
