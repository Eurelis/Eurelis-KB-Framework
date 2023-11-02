
from langchain.schema.vectorstore import VectorStore

from eurelis_kb_framework.base_factory import BaseFactory, ParamsDictFactory


class GenericVectorStoreFactory(ParamsDictFactory[VectorStore]):
    """
    Generic factory for vector store, delegate to another factory under the hood
    """
    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.provider = None

    def set_provider(self, provider: str):
        """
        Set the name of the provider to use
        Args:
            provider: name of a provider to use

        Returns:

        """
        self.provider = provider.lower()

    def _get_provider_factory(self) -> BaseFactory[VectorStore]:
        """
        Helper method to get the vector store to use
        Returns:
            a vector store factory

        """
        if not self.provider:
            raise RuntimeError('GenericEmbeddingsFactory missing provider parameter')

        if self.provider == 'chroma':
            from eurelis_kb_framework.vectorstores.chroma import ChromaFactory
            return ChromaFactory()
        elif self.provider == 'solr':
            from eurelis_kb_framework.vectorstores.solr import SolrFactory
            return SolrFactory()

        raise RuntimeError(f"GenericEmbeddingsFactory unknown provider {self.provider}")

    def build(self, context) -> VectorStore:
        """
        Method to build the vector store
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            The constructed vector store

        """
        provider_factory = self._get_provider_factory()

        context.console.verbose_print(f"Getting vectorstore from {self.provider}")

        provider_factory.set_params(self.params)

        return provider_factory.build(context)
