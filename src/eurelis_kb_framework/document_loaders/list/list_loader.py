from langchain.schema import Document

from eurelis_kb_framework.base_factory import BaseFactory
from langchain.document_loaders.base import BaseLoader

from typing import Iterable, List, Iterator


class ListLoader(BaseLoader):
    """
    List loader class to transform a single target based loader into a multi target one
    """
    def __init__(self, targets: Iterable[str], loader: BaseFactory[BaseLoader], varname: str, parameters: dict, context):
        """
        Constructor
        Args:
            targets: list of targets (strings)
            loader: parameter to get the under the hood loader factory
            varname: name of the parameter on a under the hood to use as target
            parameters: parameters for the under the hood loader factory
            context: the context object, usually the current langchain wrapper instance
        """
        self.loader = loader
        self.targets = targets
        self.varname = varname
        self.context = context
        self.parameters = parameters
        print("constructor")

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """
        Lazy load method
        Returns:
            iterator over documents
        """

        for target in self.targets:
            params = dict()
            params[self.varname] = target

            # TODO: instantiate factory only one time, and clear params during the loop
            print("ici")
            print(self.loader)
            loader_factory = self.context.loader.instantiate_factory(
                "eurelis_kb_framework.document_loaders",
                "GenericLoaderFactory",
                self.loader.copy())
            loader_factory.set_params(params)
            print(params)

            final_loader = loader_factory.build(self.context)

            try:
                # preferred method to use
                documents = final_loader.lazy_load()
            except NotImplementedError:
                # fallback if it isn't implemented
                documents = final_loader.load()

            yield from documents

    def load(self) -> List[Document]:
        """
        Load method
        Returns:
            list of documents
        """
        return list(self.lazy_load())
