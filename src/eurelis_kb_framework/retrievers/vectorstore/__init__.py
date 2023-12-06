from langchain.schema import BaseRetriever
from typing import TYPE_CHECKING

from eurelis_kb_framework import BaseFactory, T

if TYPE_CHECKING:
    from eurelis_kb_framework.langchain_wrapper import BaseContext


class VectorStoreRetrieverFactory(BaseFactory[BaseRetriever]):
    def __init__(self):
        self.retriever_kwargs = {}

    def set_retriever_kwargs(self, kwargs: dict):
        """
        Setter for vector store kwargs
        Args:
            kwargs: key value arguments to get the vector store retriever

        Returns:

        """
        self.retriever_kwargs = kwargs if kwargs else {}

    def build(self, context: "BaseContext") -> T:
        return context.vector_store.as_retriever(**self.retriever_kwargs)
