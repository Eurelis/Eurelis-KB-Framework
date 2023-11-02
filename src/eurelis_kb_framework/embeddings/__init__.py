
from langchain.schema.embeddings import Embeddings

from eurelis_kb_framework.base_factory import BaseFactory, ParamsDictFactory


class GenericEmbeddingsFactory(ParamsDictFactory[Embeddings]):
    """
    Generic embeddings factory, will delegate embeddings construction
    to another factory given a provider name
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.provider = None

    def set_provider(self, provider: str):
        """
        Setter for the provider
        Args:
            provider: name of the embeddings factory provider

        Returns:

        """
        self.provider = provider.lower()

    def _get_provider_factory(self) -> BaseFactory[Embeddings]:
        """
        Helper method to get the final factory to use
        Returns:

        """
        if not self.provider:
            raise RuntimeError('GenericEmbeddingsFactory missing provider parameter')

        if self.provider == 'openai':
            from eurelis_kb_framework.embeddings.openai import OpenAIEmbeddingsFactory
            return OpenAIEmbeddingsFactory()
        elif self.provider == 'huggingface':
            from eurelis_kb_framework.embeddings.huggingface import HuggingFaceEmbeddingsFactory
            return HuggingFaceEmbeddingsFactory()

        raise RuntimeError(f'GenericEmbeddingsFactory unknown provider {self.provider}')

    def build(self, context) -> Embeddings:
        """
        Method to construct an embeddings instance
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            embeddings instance
        """
        provider_factory = self._get_provider_factory()

        if hasattr(context, 'console'):
            context.console.verbose_print(f"Getting embeddings using {self.provider}")

        provider_factory.set_params(self.params)

        return provider_factory.build(context)
