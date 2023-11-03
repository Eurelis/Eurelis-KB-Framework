from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings

from eurelis_kb_framework.base_factory import ParamsDictFactory


class OpenAIEmbeddingsFactory(ParamsDictFactory[Embeddings]):
    """
    Factory for OpenAIEmbeddings
    """

    OPTIONAL_PARAMS = {
        "allowed_special",
        "chunk_size",
        "deployment",
        "disallowed_special",
        "embedding_ctx_length",
        "headers",
        "max_retries",
        "model",
        "model_kwargs",
        "openai_api_base",
        "openai_api_key",
        "openai_api_type",
        "openai_api_version",
        "openai_organization",
        "openai_proxy",
        "request_timeout",
        "show_progress_var",
        "skip_empty",
        "tiktoken_model_name",
    }

    def build(self, context) -> Embeddings:
        """
        Construct the openai object

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            An OpenAIEmbeddings instance
        """
        arguments = self.get_optional_params()

        return OpenAIEmbeddings(**arguments)
