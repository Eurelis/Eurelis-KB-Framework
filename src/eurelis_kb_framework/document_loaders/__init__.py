from langchain.document_loaders.base import BaseLoader

from eurelis_kb_framework.base_factory import BaseFactory, ParamsDictFactory


class GenericLoaderFactory(ParamsDictFactory[BaseLoader]):
    """
    Generic loader factory
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.provider = None

    def set_provider(self, provider: str):
        """
        Setter for the provider parameter
        Args:
            provider: name of the sub provider to use

        Returns:

        """
        self.provider = provider.lower()

    def _get_provider_factory(self) -> BaseFactory[BaseLoader]:
        """
        Helper method to get the base loader factory to use under the hood
        Returns:
            a factory for a base loader

        """
        if not self.provider:
            raise RuntimeError("GenericLoaderFactory missing 'provider' parameter")

        if self.provider == "url":
            from eurelis_kb_framework.document_loaders.url import UrlLoaderFactory

            return UrlLoaderFactory()
        elif self.provider == "fs":
            from eurelis_kb_framework.document_loaders.fs import FSLoaderFactory

            return FSLoaderFactory()
        elif self.provider == "list":
            from eurelis_kb_framework.document_loaders.list import ListLoaderFactory

            return ListLoaderFactory()

        raise RuntimeError(f"GenericEmbeddingsFactory unknown provider {self.provider}")

    def build(self, context) -> BaseLoader:
        """
        Construct the document loader

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            document loader

        """
        provider_factory = self._get_provider_factory()

        provider_factory.set_params(self.params)

        return provider_factory.build(context)
