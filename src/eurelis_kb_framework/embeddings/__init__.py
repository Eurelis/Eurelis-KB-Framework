from typing import TYPE_CHECKING

from langchain.schema.embeddings import Embeddings

from eurelis_kb_framework.base_factory import ProviderFactory

if TYPE_CHECKING:
    pass


class GenericEmbeddingsFactory(ProviderFactory[Embeddings]):
    """
    Generic embeddings factory, will delegate embeddings construction
    to another factory given a provider name
    """

    ALLOWED_PROVIDERS = {
        "openai": "eurelis_kb_framework.embeddings.openai.OpenAIEmbeddingsFactory",
        "huggingface": "eurelis_kb_framework.embeddings.huggingface.HuggingFaceEmbeddingsFactory",
    }
